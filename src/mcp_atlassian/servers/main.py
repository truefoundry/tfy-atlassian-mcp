"""Main FastMCP server setup for Atlassian integration."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional

from cachetools import TTLCache
from fastmcp import FastMCP
from fastmcp.tools import Tool as FastMCPTool
from mcp.types import Tool as MCPTool
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from mcp_atlassian.confluence import ConfluenceFetcher
from mcp_atlassian.confluence.config import ConfluenceConfig
from mcp_atlassian.jira import JiraFetcher
from mcp_atlassian.jira.config import JiraConfig
from mcp_atlassian.utils.environment import get_available_services
from mcp_atlassian.utils.io import is_read_only_mode
from mcp_atlassian.utils.logging import mask_sensitive
from mcp_atlassian.utils.tools import get_enabled_tools, should_include_tool

from .confluence import confluence_mcp
from .context import MainAppContext
from .jira import jira_mcp

logger = logging.getLogger("mcp-atlassian.server.main")


async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@asynccontextmanager
async def main_lifespan(app: FastMCP[MainAppContext]) -> AsyncIterator[dict]:
    logger.info("Main Atlassian MCP server lifespan starting...")
    services = get_available_services()
    read_only = is_read_only_mode()
    enabled_tools = get_enabled_tools()

    loaded_jira_config: JiraConfig | None = None
    loaded_confluence_config: ConfluenceConfig | None = None

    if services.get("jira"):
        try:
            jira_config = JiraConfig.from_env()
            if jira_config.is_auth_configured():
                loaded_jira_config = jira_config
                logger.info(
                    "Jira configuration loaded and authentication is configured."
                )
            else:
                logger.warning(
                    "Jira URL found, but authentication is not fully configured. "
                    "Jira tools will be unavailable."
                )
        except Exception as e:
            logger.error(f"Failed to load Jira configuration: {e}", exc_info=True)

    if services.get("confluence"):
        try:
            confluence_config = ConfluenceConfig.from_env()
            if confluence_config.is_auth_configured():
                loaded_confluence_config = confluence_config
                logger.info(
                    "Confluence configuration loaded and authentication is configured."
                )
            else:
                logger.warning(
                    "Confluence URL found, but authentication is not fully configured. "
                    "Confluence tools will be unavailable."
                )
        except Exception as e:
            logger.error(f"Failed to load Confluence configuration: {e}", exc_info=True)

    app_context = MainAppContext(
        full_jira_config=loaded_jira_config,
        full_confluence_config=loaded_confluence_config,
        read_only=read_only,
        enabled_tools=enabled_tools,
    )
    logger.info(f"Read-only mode: {'ENABLED' if read_only else 'DISABLED'}")
    logger.info(f"Enabled tools filter: {enabled_tools or 'All tools enabled'}")
    yield {"app_lifespan_context": app_context}
    logger.info("Main Atlassian MCP server lifespan shutting down.")


class AtlassianMCP(FastMCP[MainAppContext]):
    """Custom FastMCP server class for Atlassian integration with tool filtering."""

    async def _mcp_list_tools(self) -> list[MCPTool]:
        # Filter tools based on enabled_tools, read_only mode, and service
        # configuration from the lifespan context.
        req_context = self._mcp_server.request_context
        if req_context is None or req_context.lifespan_context is None:
            logger.warning(
                "Lifespan context not available during _main_mcp_list_tools call."
            )
            return []

        lifespan_ctx_dict = req_context.lifespan_context
        app_lifespan_state: MainAppContext | None = (
            lifespan_ctx_dict.get("app_lifespan_context")
            if isinstance(lifespan_ctx_dict, dict)
            else None
        )
        read_only = (
            getattr(app_lifespan_state, "read_only", False)
            if app_lifespan_state
            else False
        )
        enabled_tools_filter = (
            getattr(app_lifespan_state, "enabled_tools", None)
            if app_lifespan_state
            else None
        )
        logger.debug(
            f"_main_mcp_list_tools: read_only={read_only}, "
            f"enabled_tools_filter={enabled_tools_filter}"
        )

        all_tools: dict[str, FastMCPTool] = await self.get_tools()
        logger.debug(
            f"Aggregated {len(all_tools)} tools before filtering: "
            f"{list(all_tools.keys())}"
        )

        filtered_tools: list[MCPTool] = []
        for registered_name, tool_obj in all_tools.items():
            tool_tags = tool_obj.tags

            if not should_include_tool(registered_name, enabled_tools_filter):
                logger.debug(f"Excluding tool '{registered_name}' (not enabled)")
                continue

            if tool_obj and read_only and "write" in tool_tags:
                logger.debug(
                    f"Excluding tool '{registered_name}' due to read-only mode "
                    "and 'write' tag"
                )
                continue

            # Exclude Jira/Confluence tools if config is not fully authenticated
            is_jira_tool = "jira" in tool_tags
            is_confluence_tool = "confluence" in tool_tags
            service_configured_and_available = True
            if app_lifespan_state:
                if is_jira_tool and not app_lifespan_state.full_jira_config:
                    logger.debug(
                        f"Excluding Jira tool '{registered_name}' as Jira "
                        "configuration/authentication is incomplete."
                    )
                    service_configured_and_available = False
                if is_confluence_tool and not app_lifespan_state.full_confluence_config:
                    logger.debug(
                        f"Excluding Confluence tool '{registered_name}' as "
                        "Confluence configuration/authentication is incomplete."
                    )
                    service_configured_and_available = False
            elif is_jira_tool or is_confluence_tool:
                logger.warning(
                    f"Excluding tool '{registered_name}' as application context "
                    "is unavailable to verify service configuration."
                )
                service_configured_and_available = False

            if not service_configured_and_available:
                continue

            filtered_tools.append(tool_obj.to_mcp_tool(name=registered_name))

        logger.debug(
            f"_main_mcp_list_tools: Total tools after filtering: {len(filtered_tools)}"
        )
        return filtered_tools

    def http_app(self, *args: Any, **kwargs: Any) -> Any:
        """Override to add UserTokenMiddleware to HTTP app."""
        # Extract middleware from kwargs if it exists
        middleware_list = kwargs.get("middleware", [])
        if not isinstance(middleware_list, list):
            middleware_list = []

        # Add our middleware at the beginning
        user_token_mw = Middleware(UserTokenMiddleware, mcp_server_ref=self)
        final_middleware_list = [user_token_mw] + middleware_list
        kwargs["middleware"] = final_middleware_list

        # Call parent with updated middleware
        return super().http_app(*args, **kwargs)

    def sse_app(self, *args: Any, **kwargs: Any) -> Any:
        """Override to add UserTokenMiddleware to SSE app."""
        # Extract middleware from kwargs if it exists
        middleware_list = kwargs.get("middleware", [])
        if not isinstance(middleware_list, list):
            middleware_list = []

        # Add our middleware at the beginning
        user_token_mw = Middleware(UserTokenMiddleware, mcp_server_ref=self)
        final_middleware_list = [user_token_mw] + middleware_list
        kwargs["middleware"] = final_middleware_list

        # Call parent with updated middleware
        return super().sse_app(*args, **kwargs)


token_validation_cache: TTLCache[
    int, tuple[bool, str | None, JiraFetcher | None, ConfluenceFetcher | None]
] = TTLCache(maxsize=100, ttl=300)


class UserTokenMiddleware:
    """Middleware to extract Atlassian user tokens/credentials from
    Authorization headers."""

    def __init__(
        self, app: ASGIApp, mcp_server_ref: Optional["AtlassianMCP"] = None
    ) -> None:
        self.app = app
        self.mcp_server_ref = mcp_server_ref
        if not self.mcp_server_ref:
            logger.warning(
                "UserTokenMiddleware initialized without mcp_server_ref. "
                "Path matching for MCP endpoint might fail if settings are needed."
            )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)
        logger.debug(
            f"UserTokenMiddleware: ENTERED for request path='{request.url.path}', "
            f"method='{request.method}'"
        )

        mcp_server_instance = self.mcp_server_ref
        if mcp_server_instance is None:
            logger.debug(
                "UserTokenMiddleware: self.mcp_server_ref is None. "
                "Skipping MCP auth logic."
            )
            await self.app(scope, receive, send)
            return

        auth_header = request.headers.get("Authorization")
        token_for_log = mask_sensitive(
            auth_header.split(" ", 1)[1].strip()
            if auth_header and " " in auth_header
            else auth_header
        )
        logger.debug(
            f"UserTokenMiddleware: Path='{request.url.path}', "
            f"AuthHeader='{mask_sensitive(auth_header)}', "
            f"ParsedToken(masked)='{token_for_log}'"
        )
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if not token:
                response = JSONResponse(
                    {"error": "Unauthorized: Empty Bearer token"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return
            logger.debug(
                f"UserTokenMiddleware: Bearer token extracted (masked): "
                f"...{mask_sensitive(token, 8)}"
            )
            if "state" not in scope:
                scope["state"] = {}
            scope["state"]["user_atlassian_token"] = token
            scope["state"]["user_atlassian_auth_type"] = "oauth"
            scope["state"]["user_atlassian_email"] = None
            logger.debug(
                "UserTokenMiddleware: Set scope state (pre-validation): "
                "auth_type='oauth', token_present=True"
            )
        elif auth_header and auth_header.startswith("Token "):
            token = auth_header.split(" ", 1)[1].strip()
            if not token:
                response = JSONResponse(
                    {"error": "Unauthorized: Empty Token (PAT)"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return
            logger.debug(
                f"UserTokenMiddleware: PAT (Token scheme) extracted (masked): "
                f"...{mask_sensitive(token, 8)}"
            )
            if "state" not in scope:
                scope["state"] = {}
            scope["state"]["user_atlassian_token"] = token
            scope["state"]["user_atlassian_auth_type"] = "pat"
            scope["state"]["user_atlassian_email"] = (
                None  # PATs don't carry email in the token itself
            )
            logger.debug("UserTokenMiddleware: Set scope state for PAT auth.")
        elif auth_header:
            auth_type = (
                auth_header.split(" ", 1)[0] if " " in auth_header else "UnknownType"
            )
            logger.warning(
                f"Unsupported Authorization type for {request.url.path}: {auth_type}"
            )
            response = JSONResponse(
                {
                    "error": "Unauthorized: Only 'Bearer <OAuthToken>' or "
                    "'Token <PAT>' types are supported."
                },
                status_code=401,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
        logger.debug(
            f"UserTokenMiddleware: EXITED for request path='{request.url.path}'"
        )


main_mcp = AtlassianMCP(name="Atlassian MCP", lifespan=main_lifespan)
main_mcp.mount("jira", jira_mcp)
main_mcp.mount("confluence", confluence_mcp)


@main_mcp.custom_route("/healthz", methods=["GET"], include_in_schema=False)
async def _health_check_route(request: Request) -> JSONResponse:
    return await health_check(request)


logger.info("Added /healthz endpoint for Kubernetes probes")

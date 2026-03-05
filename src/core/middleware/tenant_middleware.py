"""
Tenant context middleware

Extracts tenant ID from HTTP request headers and populates the tenant context
for the duration of the request. This enables multi-tenant mode where each
request is scoped to a specific tenant.

This middleware should only be registered when multi-tenant mode is active
(TENANT_NON_TENANT_MODE=false and TENANT_SINGLE_TENANT_ID is not set).
"""

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable

from core.tenants.tenant_contextvar import set_current_tenant, clear_current_tenant
from core.tenants.tenant_config import get_tenant_config
from core.tenants.tenant_info_provider import TenantInfoService
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Tenant context middleware

    Extracts tenant ID from a configurable HTTP header (default: X-Tenant-Id),
    resolves tenant info via TenantInfoService, and sets it into the ContextVar
    for the duration of the request.

    Lifecycle:
    1. Read tenant ID from request header
    2. Resolve full TenantInfo via TenantInfoService
    3. Call set_current_tenant() to populate the ContextVar
    4. Execute the rest of the middleware chain / route handler
    5. In finally: call clear_current_tenant() to clean up
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._tenant_config = get_tenant_config()
        self._tenant_info_service = get_bean_by_type(TenantInfoService)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        tenant_id = self._extract_tenant_id(request)

        if not tenant_id:
            if self._tenant_config.app_ready:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required header: {self._tenant_config.tenant_header_name}",
                )
            else:
                logger.warning(
                    "No tenant ID in request header '%s' during startup, proceeding without tenant context",
                    self._tenant_config.tenant_header_name,
                )
                return await call_next(request)

        tenant_info = self._tenant_info_service.get_tenant_info(tenant_id)
        if tenant_info is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tenant: {tenant_id}",
            )

        set_current_tenant(tenant_info)
        try:
            response = await call_next(request)
            return response
        finally:
            clear_current_tenant()
            logger.debug("Tenant context cleaned up for tenant: %s", tenant_id)

    def _extract_tenant_id(self, request: Request) -> str | None:
        """
        Extract tenant ID from the configured HTTP header.

        Tries the configured header name first, then falls back to a
        case-insensitive lookup (HTTP headers are case-insensitive per RFC 7230).

        Args:
            request: FastAPI request object

        Returns:
            Tenant ID string, or None if not present
        """
        header_name = self._tenant_config.tenant_header_name
        tenant_id = request.headers.get(header_name)
        if tenant_id:
            return tenant_id.strip()

        # Starlette headers are case-insensitive, but try lowercase explicitly
        tenant_id = request.headers.get(header_name.lower())
        if tenant_id:
            return tenant_id.strip()

        return None

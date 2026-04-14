"""
URL analysis service — fetches content from URLs and routes to detectors.
Includes SSRF protection to block access to internal/private networks.
"""

from __future__ import annotations

import ipaddress
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from ..models.models import ContentType

log = structlog.get_logger(__name__)

# Size limits per content type (bytes)
MAX_SIZES = {
    ContentType.TEXT:  10 * 1024 * 1024,   # 10 MB
    ContentType.IMAGE: 50 * 1024 * 1024,   # 50 MB
    ContentType.AUDIO: 200 * 1024 * 1024,  # 200 MB
    ContentType.VIDEO: 500 * 1024 * 1024,  # 500 MB
    ContentType.CODE:  5 * 1024 * 1024,    # 5 MB
}

FETCH_TIMEOUT = 30.0
MAX_REDIRECTS = 5

# Content-Type header → ContentType mapping
MIME_MAP: dict[str, ContentType] = {
    "image/jpeg":      ContentType.IMAGE,
    "image/png":       ContentType.IMAGE,
    "image/webp":      ContentType.IMAGE,
    "image/gif":       ContentType.IMAGE,
    "image/bmp":       ContentType.IMAGE,
    "image/tiff":      ContentType.IMAGE,
    "video/mp4":       ContentType.VIDEO,
    "video/quicktime": ContentType.VIDEO,
    "video/x-msvideo": ContentType.VIDEO,
    "video/webm":      ContentType.VIDEO,
    "video/x-matroska": ContentType.VIDEO,
    "audio/mpeg":      ContentType.AUDIO,
    "audio/wav":       ContentType.AUDIO,
    "audio/x-wav":     ContentType.AUDIO,
    "audio/flac":      ContentType.AUDIO,
    "audio/ogg":       ContentType.AUDIO,
    "audio/mp4":       ContentType.AUDIO,
    "application/pdf": ContentType.TEXT,
    "text/plain":      ContentType.TEXT,
}

# Private/reserved IP ranges that must be blocked (SSRF protection)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def validate_url(url: str) -> str:
    """Validate and normalize URL. Raises ValueError on invalid input."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use http or https scheme")

    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    if parsed.port and parsed.port not in (80, 443, 8080, 8443):
        raise ValueError("Non-standard ports are not allowed for security")

    # Resolve hostname and check against private IP ranges
    try:
        resolved = socket.getaddrinfo(parsed.hostname, None)
        for _, _, _, _, sockaddr in resolved:
            ip = ipaddress.ip_address(sockaddr[0])
            for network in _BLOCKED_NETWORKS:
                if ip in network:
                    raise ValueError(f"Access to private/internal networks is blocked")
    except socket.gaierror:
        raise ValueError(f"Could not resolve hostname: {parsed.hostname}")

    return url


def _detect_content_type(response: httpx.Response) -> ContentType:
    """Determine content type from HTTP response headers."""
    ct = response.headers.get("content-type", "").split(";")[0].strip().lower()

    if ct in MIME_MAP:
        return MIME_MAP[ct]

    if ct.startswith("image/"):
        return ContentType.IMAGE
    if ct.startswith("video/"):
        return ContentType.VIDEO
    if ct.startswith("audio/"):
        return ContentType.AUDIO
    if ct.startswith("text/"):
        return ContentType.TEXT

    # Default: treat as text (HTML pages, etc.)
    return ContentType.TEXT


def _extract_text_from_html(html: str) -> str:
    """
    Extract readable article text from HTML.
    Uses simple tag stripping as a baseline — production would use
    readability-lxml or trafilatura for better extraction.
    """
    import re

    # Remove script and style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Decode HTML entities
    import html as html_module
    text = html_module.unescape(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


async def fetch_and_analyze_url(url: str) -> tuple[ContentType, bytes | str, dict[str, Any]]:
    """
    Fetch content from a URL and determine its type for detection.

    Returns:
        (content_type, content, metadata)
        - content is str for text, bytes for binary content
        - metadata includes source_url, domain, fetch_timestamp, content_length

    Raises:
        ValueError: On invalid URL, SSRF attempt, or oversized content
        httpx.HTTPError: On network errors
    """
    url = validate_url(url)
    parsed = urlparse(url)

    metadata: dict[str, Any] = {
        "source_url":      url,
        "domain":          parsed.hostname,
        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Hard cap: never download more than the largest allowed size
    absolute_max = max(MAX_SIZES.values())

    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT,
        follow_redirects=False,
    ) as client:
        # First pass: follow redirects with SSRF validation on each hop
        target_url = url
        redirects_followed = 0
        while True:
            response = await client.head(target_url)
            if not response.is_redirect:
                break
            if redirects_followed >= MAX_REDIRECTS:
                raise ValueError(f"Too many redirects (>{MAX_REDIRECTS})")
            location = response.headers.get("location")
            if not location:
                break
            if not location.startswith(("http://", "https://")):
                from urllib.parse import urljoin
                location = urljoin(str(response.url), location)
            target_url = validate_url(location)
            redirects_followed += 1

        # Check Content-Length header before downloading body
        declared_length = response.headers.get("content-length")
        if declared_length and int(declared_length) > absolute_max:
            raise ValueError(
                f"Content too large: server declared {int(declared_length) / 1024 / 1024:.1f} MB "
                f"(absolute max {absolute_max / 1024 / 1024:.0f} MB)"
            )

        # Stream the actual GET with incremental size enforcement
        chunks: list[bytes] = []
        downloaded = 0
        async with client.stream("GET", target_url) as stream:
            stream.raise_for_status()
            async for chunk in stream.aiter_bytes(chunk_size=64 * 1024):
                downloaded += len(chunk)
                if downloaded > absolute_max:
                    raise ValueError(
                        f"Content too large: exceeded {absolute_max / 1024 / 1024:.0f} MB "
                        f"during download (streamed)"
                    )
                chunks.append(chunk)
            # Capture headers from the streamed response
            response = stream

    content_bytes = b"".join(chunks)
    content_length = len(content_bytes)

    content_type = _detect_content_type(response)
    metadata["content_length"] = content_length
    metadata["content_type_header"] = response.headers.get("content-type", "")
    metadata["status_code"] = response.status_code

    # Check per-type size limits
    max_size = MAX_SIZES.get(content_type, MAX_SIZES[ContentType.TEXT])
    if content_length > max_size:
        raise ValueError(
            f"Content too large: {content_length / 1024 / 1024:.1f} MB "
            f"(max {max_size / 1024 / 1024:.0f} MB for {content_type.value})"
        )

    # Extract content based on type
    if content_type == ContentType.TEXT:
        ct_header = response.headers.get("content-type", "")
        text_str = content_bytes.decode("utf-8", errors="replace")
        if "html" in ct_header.lower():
            text_str = _extract_text_from_html(text_str)

        if len(text_str.strip()) < 20:
            raise ValueError("Extracted text is too short to analyze (minimum 20 characters)")

        log.info("url_fetched", url=url, type="text", chars=len(text_str))
        return content_type, text_str, metadata
    else:
        log.info("url_fetched", url=url, type=content_type.value, bytes=content_length)
        return content_type, content_bytes, metadata

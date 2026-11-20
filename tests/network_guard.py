"""Block outbound internet access during pytest.

The suite is supposed to run from checked-in files only. A socket-level
patch is not enough: yfinance 1.8 prefers curl_cffi (libcurl), which never
calls Python's socket module. This helper also wraps curl_cffi, requests
and urllib so a stray download fails instead of quietly hitting Yahoo.
Loopback and Unix sockets stay allowed.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse


class NetworkDisabledError(RuntimeError):
    """A test tried to open an outbound internet connection."""


_REAL_SOCKET = socket.socket
_REAL_CREATE_CONNECTION = socket.create_connection
_REAL_GETADDRINFO = socket.getaddrinfo
_REAL_URLOPEN = urllib.request.urlopen

_restore: list[tuple[Any, str, Any]] = []


def _is_loopback(host: object) -> bool:
    if host is None:
        return False
    if isinstance(host, bytes):
        try:
            host = host.decode()
        except UnicodeDecodeError:
            return False
    text = str(host).strip("[]")
    if text in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def reject(target: object) -> None:
    raise NetworkDisabledError(
        f"Network access is disabled in tests (tried to reach {target!r}). "
        "Use local fixtures or checked-in data; do not download."
    )


def _unix_address(address: object) -> bool:
    return isinstance(address, (str, bytes)) and not isinstance(address, tuple)


class _GuardedSocket(_REAL_SOCKET):
    """Allow Unix sockets and loopback; block outbound TCP/UDP connects."""

    def connect(self, address):
        if self.family == socket.AF_UNIX or _unix_address(address):
            return super().connect(address)
        host = address[0] if isinstance(address, tuple) else address
        if not _is_loopback(host):
            reject(host)
        return super().connect(address)

    def connect_ex(self, address):
        if self.family == socket.AF_UNIX or _unix_address(address):
            return super().connect_ex(address)
        host = address[0] if isinstance(address, tuple) else address
        if not _is_loopback(host):
            reject(host)
        return super().connect_ex(address)


def _guarded_create_connection(address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    if not _is_loopback(host):
        reject(host)
    return _REAL_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    if host is None or _is_loopback(host):
        return _REAL_GETADDRINFO(host, port, *args, **kwargs)
    try:
        ipaddress.ip_address(str(host).strip("[]"))
    except ValueError:
        reject(host)
    return _REAL_GETADDRINFO(host, port, *args, **kwargs)


def _host_from_url(url: object) -> object:
    if not isinstance(url, str):
        return url
    return urlparse(url).hostname or url


def _guarded_urlopen(url, *args, **kwargs):
    target = url.full_url if hasattr(url, "full_url") else url
    host = _host_from_url(str(target))
    if not _is_loopback(host):
        reject(target)
    return _REAL_URLOPEN(url, *args, **kwargs)


def _patch(obj: Any, name: str, replacement: Any) -> None:
    _restore.append((obj, name, getattr(obj, name)))
    setattr(obj, name, replacement)


def _wrap_request(original: Callable[..., Any]) -> Callable[..., Any]:
    def _request(self, method, url, *args, **kwargs):
        host = _host_from_url(url)
        if not _is_loopback(host):
            reject(url)
        return original(self, method, url, *args, **kwargs)

    return _request


def _wrap_module_request(original: Callable[..., Any]) -> Callable[..., Any]:
    def _request(method, url, *args, **kwargs):
        host = _host_from_url(url)
        if not _is_loopback(host):
            reject(url)
        return original(method, url, *args, **kwargs)

    return _request


def _patch_http_backends() -> None:
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        _patch(requests.sessions.Session, "request", _wrap_request(requests.sessions.Session.request))
        if hasattr(requests, "api") and hasattr(requests.api, "request"):
            _patch(requests.api, "request", _wrap_module_request(requests.api.request))

    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        curl_requests = None
    if curl_requests is not None:
        session_cls = curl_requests.Session
        _patch(session_cls, "request", _wrap_request(session_cls.request))
        if hasattr(curl_requests, "request"):
            _patch(curl_requests, "request", _wrap_module_request(curl_requests.request))


def install() -> None:
    """Enable the process-wide outbound-network block."""
    if _restore:
        return
    _patch_http_backends()
    socket.socket = _GuardedSocket
    socket.create_connection = _guarded_create_connection
    socket.getaddrinfo = _guarded_getaddrinfo
    urllib.request.urlopen = _guarded_urlopen


def uninstall() -> None:
    socket.socket = _REAL_SOCKET
    socket.create_connection = _REAL_CREATE_CONNECTION
    socket.getaddrinfo = _REAL_GETADDRINFO
    urllib.request.urlopen = _REAL_URLOPEN
    while _restore:
        obj, name, original = _restore.pop()
        setattr(obj, name, original)

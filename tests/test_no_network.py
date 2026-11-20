"""Pytest must not reach the internet; only the data loader may download."""

from __future__ import annotations

import socket

import pytest
import requests

from tests.network_guard import NetworkDisabledError


def test_getaddrinfo_blocks_public_hostnames() -> None:
    with pytest.raises(NetworkDisabledError, match="example.invalid"):
        socket.getaddrinfo("example.invalid", 443)


def test_connect_blocks_public_ip() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkDisabledError, match="8.8.8.8"):
            sock.connect(("8.8.8.8", 53))
    finally:
        sock.close()


def test_create_connection_blocks_public_ip() -> None:
    with pytest.raises(NetworkDisabledError, match="1.1.1.1"):
        socket.create_connection(("1.1.1.1", 443), timeout=0.1)


def test_loopback_name_lookup_is_still_allowed() -> None:
    answers = socket.getaddrinfo("localhost", None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    assert answers


def test_requests_get_is_blocked() -> None:
    with pytest.raises(NetworkDisabledError, match="example.invalid"):
        requests.get("https://example.invalid/", timeout=1)


def test_curl_cffi_get_is_blocked() -> None:
    from curl_cffi import requests as curl_requests

    with pytest.raises(NetworkDisabledError, match="example.invalid"):
        curl_requests.get("https://example.invalid/", timeout=1)

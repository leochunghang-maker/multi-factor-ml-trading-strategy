"""The platform is local simulated paper trading only: no broker client may creep in.

Static AST/text scan of ``src/``, ``scripts/`` and ``dashboard/``, no code is run.
Fails if any module imports a broker SDK or a general HTTP client, or names a
broker/order endpoint. Prices come from yfinance in ``src/data_loader.py`` only; that
module may also use ``urllib.request`` to refresh the paper-local membership file.
These are the only allowed network dependencies; orders are filled by the local
simulated broker and never leave the machine. Other modules must use the local files
that loader writes, never import yfinance or an HTTP client themselves.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED_DIRS = ("src", "scripts", "dashboard")

# Top-level import roots (or dotted prefixes) that would let code place orders or
# call an arbitrary HTTP API.
FORBIDDEN_IMPORTS = (
    "alpaca",
    "alpaca_trade_api",
    "ib_insync",
    "ibapi",
    "ccxt",
    "tda",
    "schwab",
    "robin_stocks",
    "oandapyV20",
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    "websocket",
    "websockets",
    "urllib.request",
    "http.client",
)
FORBIDDEN_TEXT = re.compile(
    r"alpaca\.markets|ALPACA_API_KEY|APCA-API|interactivebrokers|paper-api\.|/v2/orders",
    re.IGNORECASE,
)


def _python_files() -> list[Path]:
    return sorted(path for name in SCANNED_DIRS for path in (ROOT / name).rglob("*.py"))


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _is_forbidden(module: str) -> bool:
    return any(module == name or module.startswith(name + ".") for name in FORBIDDEN_IMPORTS)


def test_scan_covers_the_code() -> None:
    files = {str(path.relative_to(ROOT)) for path in _python_files()}
    assert "src/execution/simulated_broker.py" in files
    assert "scripts/run_daily_paper_trading.py" in files
    assert "dashboard/app.py" in files


# urllib.request is allowed only in the data loader (paper membership refresh).
ALLOWED_HTTP_IMPORTS = {
    "src/data_loader.py": {"urllib.request"},
}


def test_no_broker_sdk_or_http_client_imports() -> None:
    offenders = []
    for path in _python_files():
        relative = str(path.relative_to(ROOT))
        allowed = ALLOWED_HTTP_IMPORTS.get(relative, set())
        for module in sorted(_imported_modules(ast.parse(path.read_text()))):
            if _is_forbidden(module) and module not in allowed:
                offenders.append(f"{relative}: {module}")
    assert not offenders, "broker/HTTP client imports in a local-only platform:\n" + "\n".join(offenders)


def test_no_broker_endpoints_or_credentials_named() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in _python_files()
        if FORBIDDEN_TEXT.search(path.read_text())
    ]
    assert not offenders, f"broker endpoint or credential names found in: {offenders}"


ALLOWED_YFINANCE_IMPORTS = {"src/data_loader.py"}


def test_yfinance_is_imported_only_in_the_data_loader() -> None:
    offenders = []
    allowed_seen = False
    for path in _python_files():
        modules = _imported_modules(ast.parse(path.read_text()))
        if not any(module == "yfinance" or module.startswith("yfinance.") for module in modules):
            continue
        relative = str(path.relative_to(ROOT))
        if relative in ALLOWED_YFINANCE_IMPORTS:
            allowed_seen = True
            continue
        offenders.append(relative)
    assert allowed_seen, "src/data_loader.py should still import yfinance for price refresh"
    assert not offenders, "yfinance must stay in src/data_loader.py only:\n" + "\n".join(offenders)


def test_detector_flags_broker_code() -> None:
    sample = ast.parse(
        "import requests\nfrom urllib.request import urlopen\nfrom alpaca.trading import TradingClient\n"
        "from urllib import request\nimport urllib.parse\n"
    )
    flagged = {module for module in _imported_modules(sample) if _is_forbidden(module)}
    assert flagged == {
        "requests",
        "urllib.request",
        "urllib.request.urlopen",
        "alpaca.trading",
        "alpaca.trading.TradingClient",
    }
    assert FORBIDDEN_TEXT.search('BASE = "https://paper-api.alpaca.markets/v2"')
    assert not FORBIDDEN_TEXT.search("local simulated broker")


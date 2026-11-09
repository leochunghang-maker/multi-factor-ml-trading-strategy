"""Config keys read in src/ must exist in config/platform_config.json, and vice versa.

Static AST scan, no code is run. A read is any ``<section>["key"]`` or
``<section>.get("key")`` where ``<section>`` is a top-level config section
(``config["paper_trading"]``, ``config.get("research") or {}``, or a name
assigned from one, e.g. ``_RISK = _PLATFORM.get("risk_limits", {})``).
Sections merged over a module default (``{**DEFAULT_X, **config.get("sec")}``)
count every default key as read, and the default must list only keys that the
config file has. Catches typos in either place, which otherwise fall back to a
silent code default.
"""

import ast
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config/platform_config.json").read_text())
SECTIONS = {name for name, value in CONFIG.items() if isinstance(value, dict)}
# Descriptive keys documented for humans and never read by code.
DESCRIPTIVE = {("strategy", "universe_name"), ("execution", "mode")}


def _literal(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _section(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return _section(node.values[0], aliases)
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Subscript):
        key = _literal(node.slice)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
        key = _literal(node.args[0])
    else:
        return None
    return key if key in SECTIONS else None


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def scan_config_reads() -> tuple[set[tuple[str, str, str]], dict[tuple[str, str], set[str]]]:
    """Return {(section, key, file)} reads and {(module, DEFAULT name): section} merges."""
    reads: set[tuple[str, str, str]] = set()
    merges: dict[tuple[str, str], set[str]] = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text())
        rel = str(path.relative_to(ROOT))
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                section = _section(node.value, {})
                if section:
                    aliases[node.targets[0].id] = section
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and None in node.keys:
                unpacked = [value for key, value in zip(node.keys, node.values) if key is None]
                sections = {s for s in (_section(v, aliases) for v in unpacked) if s}
                defaults = [v.id for v in unpacked if isinstance(v, ast.Name) and v.id.startswith("DEFAULT")]
                for name in defaults:
                    merges.setdefault((_module_name(path), name), set()).update(sections)
                continue
            if isinstance(node, ast.Subscript):
                base, key = node.value, _literal(node.slice)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
                base, key = node.func.value, _literal(node.args[0])
            else:
                continue
            section = _section(base, aliases) if key else None
            if section:
                reads.add((section, key, rel))
    return reads, merges


def _merged_defaults() -> dict[str, set[str]]:
    _, merges = scan_config_reads()
    keys: dict[str, set[str]] = {}
    for (module, name), sections in merges.items():
        default = getattr(importlib.import_module(module), name)
        for section in sections:
            keys.setdefault(section, set()).update(default)
    return keys


def test_scan_finds_reads_in_config_module_and_operations() -> None:
    reads, merges = scan_config_reads()
    found = {(section, key) for section, key, _ in reads}
    assert ("risk_limits", "max_positions") in found  # src/config.py alias
    assert ("paper_trading", "data_dir") in found  # (config.get(...) or {}).get(...)
    assert merges
    assert {"data_checks", "paper_trading", "paper_tracking"} <= set(_merged_defaults())


def test_every_key_read_in_src_exists_in_platform_config() -> None:
    reads, _ = scan_config_reads()
    missing = sorted(f"{file}: {section}.{key}" for section, key, file in reads if key not in CONFIG[section])
    assert not missing, "config keys read in src/ but absent from platform_config.json:\n" + "\n".join(missing)


def test_module_defaults_only_name_keys_the_config_file_has() -> None:
    extra = sorted(
        f"{section}.{key}"
        for section, keys in _merged_defaults().items()
        for key in keys
        if key not in CONFIG[section]
    )
    assert not extra, "module defaults with no platform_config.json entry:\n" + "\n".join(extra)


def test_every_config_key_is_read_somewhere_in_src() -> None:
    reads, _ = scan_config_reads()
    read = {(section, key) for section, key, _ in reads}
    read |= {(section, key) for section, keys in _merged_defaults().items() for key in keys}
    unused = sorted(
        f"{section}.{key}"
        for section in SECTIONS
        for key in CONFIG[section]
        if (section, key) not in read and (section, key) not in DESCRIPTIVE and not key.endswith("note")
    )
    assert not unused, "platform_config.json keys never read in src/ (typo or dead knob):\n" + "\n".join(unused)


def test_execution_mode_stays_local_simulated_paper_only() -> None:
    assert CONFIG["execution"]["mode"] == "local_simulated_paper_only"

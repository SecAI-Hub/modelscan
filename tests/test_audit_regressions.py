"""Synthetic pickle fixtures are disassembled only; never unpickle them."""

import io
import json
import pickle
import struct
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from modelscan.cli import cli
from modelscan.model import Model
from modelscan.modelscan import ModelScan
from modelscan.settings import DEFAULT_SETTINGS, SettingsUtils
from modelscan.tools.picklescanner import GenOpsError, _list_globals, scan_pickle_bytes


def test_current_directory_cannot_select_executable_plugins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    model = tmp_path / "safe.pkl"
    model.write_bytes(pickle.dumps({"fixture": [1, 2]}))
    marker = tmp_path / "plugin-executed"
    (tmp_path / "audit_plugin.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "class Plugin: pass\n"
    )
    settings = SettingsUtils.get_default_settings_as_toml()
    settings += '\n[middlewares."audit_plugin.Plugin"]\n'
    (tmp_path / "modelscan-settings.toml").write_text(settings)
    result = CliRunner().invoke(cli, ["-p", str(model), "-r", "json"])
    assert result.exit_code == 0, result.output
    assert not marker.exists()
    assert "built-in settings" in result.output
    # Explicitly selecting executable configuration still exercises the
    # documented extension boundary, even if this dummy plugin cannot scan.
    CliRunner().invoke(
        cli, ["-p", str(model), "--settings-file", "modelscan-settings.toml"]
    )
    assert marker.exists()
    sys.modules.pop("audit_plugin", None)


@pytest.mark.parametrize(
    "module,name",
    [
        ("ctypes", "CDLL"),
        ("ctypes", "PyDLL"),
        ("_ctypes", "dlopen"),
        ("numpy.ctypeslib", "load_library"),
        ("importlib", "import_module"),
    ],
)
def test_dynamic_native_loading_is_reported_without_execution(tmp_path, module, name):
    p = tmp_path / "native.pkl"
    p.write_bytes(f"c{module}\n{name}\n.".encode())
    report = ModelScan().scan(str(p))
    assert report["issues"]


@pytest.mark.parametrize("prefix", [b"", b"cbuiltins\nset\n.", b"cos\nsystem\n."])
def test_every_partial_stream_preserves_parse_errors(prefix):
    model = Model(Path("partial.pkl"), io.BytesIO(prefix + b"\x00"))
    result = scan_pickle_bytes(model, DEFAULT_SETTINGS)
    assert result.errors
    if b"system" in prefix:
        assert result.issues


def test_attacker_declared_argument_is_rejected_before_allocation():
    class CheckedStream(io.BytesIO):
        def read(self, size=-1):
            assert size <= 1024 * 1024, "attempted attacker-sized allocation"
            return super().read(size)

    stream = CheckedStream(b"\x80\x04\x8e" + struct.pack("<Q", 2**50))
    with pytest.raises(GenOpsError, match="argument limit"):
        _list_globals(stream)


def test_opcode_limit_bounds_an_operation_dense_small_file():
    with pytest.raises(GenOpsError, match="opcode limit"):
        _list_globals(
            io.BytesIO(b"N" * 1000 + b"."),
            settings={"scan": {"max_pickle_opcodes": 10}},
        )


def test_memo_limit_is_enforced():
    data = b"\x80\x04" + b"N\x94" * 20 + b"."
    with pytest.raises(GenOpsError, match="memo limit"):
        _list_globals(
            io.BytesIO(data), settings={"scan": {"max_pickle_memo_entries": 3}}
        )


def test_regular_multiple_pickle_streams_still_scan():
    data = pickle.dumps({"a": 1}) + pickle.dumps({"b": 2})
    result = scan_pickle_bytes(
        Model(Path("complete.pkl"), io.BytesIO(data)), DEFAULT_SETTINGS
    )
    assert not result.issues and not result.errors


def test_generated_settings_round_trip(tmp_path):
    settings = tmp_path / "trusted.toml"
    settings.write_text(SettingsUtils.get_default_settings_as_toml())
    model = tmp_path / "safe.pkl"
    model.write_bytes(pickle.dumps({"fixture": [1, 2]}))
    report_path = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli,
        [
            "-p",
            str(model),
            "--settings-file",
            str(settings),
            "-r",
            "json",
            "-o",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    assert report["summary"]["scanned"] == {
        "total_scanned": 1,
        "scanned_files": ["safe.pkl"],
    }
    assert report["issues"] == report["errors"] == []


def test_report_retains_findings_and_errors_from_one_partial_stream(tmp_path):
    path = tmp_path / "partial.pkl"
    path.write_bytes(b"cos\nsystem\n\x00")
    report = ModelScan().scan(str(path))
    assert report["issues"] and report["errors"]
    assert report["summary"]["scanned"]["total_scanned"] == 0

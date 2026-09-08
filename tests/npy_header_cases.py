"""Shared bounded NPY cases for pytest and isolated installed-NumPy probes.

Pickle bytes are disassembled only. No NumPy/pickle load operation is used.
"""

import copy
import io
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Tuple

import numpy as np

from modelscan.issues import OperatorIssueDetails
from modelscan.model import Model
from modelscan.modelscan import ModelScan
from modelscan.settings import DEFAULT_SETTINGS, Property
from modelscan.tools.picklescanner import _read_numpy_array_header, scan_numpy

VERSIONS = ((1, 0), (2, 0), (3, 0))


def make_npy(
    version: Tuple[int, int],
    *,
    descr: Any = "<i4",
    shape: Any = (1,),
    fortran: Any = False,
    text: Any = None,
    payload: bytes = b"",
) -> Tuple[bytes, int]:
    if text is None:
        text = repr({"descr": descr, "fortran_order": fortran, "shape": shape})
    header = (
        text
        if isinstance(text, bytes)
        else text.encode("utf-8" if version == (3, 0) else "latin1")
    )
    fmt = "<H" if version == (1, 0) else "<I"
    header += b" " * (-(8 + struct.calcsize(fmt) + len(header) + 1) % 64) + b"\n"
    prefix = b"\x93NUMPY" + bytes(version) + struct.pack(fmt, len(header))
    return prefix + header + payload, len(prefix) + len(header)


class CheckedStream(io.BytesIO):
    def __init__(
        self,
        data: bytes,
        *,
        chunk: Optional[int] = None,
        boundary: Optional[int] = None,
    ) -> None:
        super().__init__(data)
        self.chunk = chunk
        self.boundary = boundary
        self.requests: List[Tuple[int, int]] = []

    def read(self, size: Any = -1) -> bytes:
        self.requests.append((self.tell(), size))
        assert 0 <= size <= 10_000, "unbounded or oversized original-stream read"
        if self.boundary is not None:
            assert self.tell() + size <= self.boundary, "numeric payload was read"
        return super().read(min(size, self.chunk) if self.chunk else size)


class OffsetModel(Model):
    def __init__(self, data: IO[bytes]) -> None:
        super().__init__(Path("fixture.npy"), data)
        self.offsets: List[int] = []

    def get_stream(self, offset: int = 0) -> IO[bytes]:
        self.offsets.append(offset)
        return super().get_stream(offset)


def settings_snapshot(value: Any) -> Any:
    # SettingsUtils serializes Property keys by name; deepcopy creates new
    # identity keys, so raw dictionary equality is not a mutation check.
    if isinstance(value, dict):
        return {
            key.name if isinstance(key, Property) else key: settings_snapshot(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [settings_snapshot(item) for item in value]
    return value


CASES: List[Dict[str, Any]] = []
for version in VERSIONS:
    prefix = f"v{version[0]}"
    for variant in (
        "scalar",
        "zero",
        "fortran",
        "unicode",
        "nested-object",
        "short-read",
        "numeric-boundary",
    ):
        CASES.append(
            {
                "name": f"{prefix}-{variant}",
                "kind": "header",
                "version": version,
                "variant": variant,
            }
        )
    for variant in (
        "empty-length",
        "oversized-length",
        "truncated-length",
        "truncated-header",
        "shape-bool",
        "shape-negative",
        "fortran-int",
    ):
        CASES.append(
            {
                "name": f"{prefix}-{variant}",
                "kind": "bad-header",
                "version": version,
                "variant": variant,
            }
        )
    for variant in (
        "safe",
        "unsafe",
        "partial",
        "argument-cap",
        "opcode-cap",
        "memo-cap",
        "byte-cap",
    ):
        CASES.append(
            {
                "name": f"{prefix}-object-{variant}",
                "kind": "object",
                "version": version,
                "variant": variant,
            }
        )
for variant in (
    "duplicate",
    "extra",
    "missing",
    "not-dict",
    "shape-list",
    "dtype-none",
    "dtype-dict",
    "bad-dtype",
    "expression",
    "invalid-utf8",
    "deep-recursion",
):
    CASES.append(
        {
            "name": f"v3-{variant}",
            "kind": "v3-invalid",
            "version": (3, 0),
            "variant": variant,
        }
    )
for version in ((0, 0), (1, 1), (2, 1), (3, 1), (4, 0), (255, 255)):
    CASES.append(
        {"name": f"unsupported-{version}", "kind": "unsupported", "version": version}
    )
for variant in ("argument", "bytes", "invalid-setting"):
    CASES.append(
        {
            "name": f"trusted-header-{variant}",
            "kind": "header-limit",
            "variant": variant,
        }
    )
for version in VERSIONS:
    for variant in ("numeric", "object"):
        CASES.append(
            {
                "name": f"public-writer-v{version[0]}-{variant}",
                "kind": "public-writer",
                "version": version,
                "variant": variant,
            }
        )
for variant in (
    "nonseek",
    "raw-safe",
    "raw-unsafe",
    "zip",
    "integration-header",
    "integration-object",
    "truncated-magic",
    "truncated-version",
):
    CASES.append({"name": variant, "kind": "boundary", "variant": variant})


def expect_header_error(
    stream: IO[bytes],
    version: Tuple[int, int],
    settings: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        _read_numpy_array_header(stream, version, settings)
    except ValueError:
        return
    raise AssertionError("malformed header was accepted")


def run_case(case: Dict[str, Any]) -> None:
    kind = case["kind"]
    version = case.get("version", (3, 0))
    variant: str = case.get("variant", "")
    if kind == "public-writer":
        field = "雪☃" if version == (3, 0) else "café"
        dtype = (
            np.dtype([(field, [(field, object)])])
            if variant == "object"
            else np.dtype([(field, "<i4")])
        )
        array = np.zeros((1,), dtype=dtype)
        output = io.BytesIO()
        # Writing serializes a benign array; no pickle loader is ever invoked.
        np.lib.format.write_array(
            output, array, version=version, allow_pickle=variant == "object"
        )
        data = output.getvalue()
        assert data[:8] == b"\x93NUMPY" + bytes(version)
        stream = CheckedStream(data)
        stream.seek(8)
        shape, fortran, actual_dtype = _read_numpy_array_header(stream, version)
        end = stream.tell()
        assert shape == (1,) and fortran is False and actual_dtype == dtype
        assert actual_dtype.names == (field,)
        model = OffsetModel(stream)
        result = scan_numpy(model, DEFAULT_SETTINGS)
        assert not result.issues and not result.errors and not result.skipped
        if variant == "object":
            assert end in model.offsets
        return
    if kind == "header":
        shape, fortran = (1,), False
        descr: Any = "<i4"
        field = "雪☃" if version == (3, 0) else "café"
        if variant == "scalar":
            shape = ()
        elif variant == "zero":
            shape = (0, 2)
        elif variant == "fortran":
            shape, fortran = (2, 3), True
        elif variant in ("unicode", "nested-object"):
            descr = (
                [(field, [(field, "|O")])]
                if variant == "nested-object"
                else [(field, "<i4")]
            )
        data, end = make_npy(
            version,
            descr=descr,
            shape=shape,
            fortran=fortran,
            payload=b"DO NOT READ NUMERIC DATA",
        )
        stream = CheckedStream(
            data, chunk=1 if variant == "short-read" else None, boundary=end
        )
        stream.seek(8)
        actual_shape, actual_fortran, dtype = _read_numpy_array_header(stream, version)
        assert actual_shape == shape and actual_fortran is fortran
        assert dtype.hasobject == (variant == "nested-object")
        if variant in ("unicode", "nested-object"):
            assert dtype.names == (
                field,
            ), "Unicode must not become UTF-8-as-Latin1 mojibake"
        if variant == "nested-object":
            assert dtype.fields[field][0].names == (field,)
        assert stream.tell() == end
        if not dtype.hasobject:
            result = scan_numpy(OffsetModel(stream), DEFAULT_SETTINGS)
            assert not result.issues and not result.errors and not result.skipped
            assert stream.tell() == end
        return
    if kind == "bad-header":
        data, end = make_npy(version)
        size = 2 if version == (1, 0) else 4
        if variant in ("empty-length", "oversized-length"):
            amount = (
                0
                if variant == "empty-length"
                else (65535 if version == (1, 0) else 2**32 - 1)
            )
            data = data[:8] + struct.pack("<H" if size == 2 else "<I", amount)
        elif variant == "truncated-length":
            data = data[: 8 + size - 1]
        elif variant == "truncated-header":
            data = data[: end - 1]
        elif variant == "shape-bool":
            data, _ = make_npy(version, shape=(True,))
        elif variant == "shape-negative":
            data, _ = make_npy(version, shape=(-1,))
        elif variant == "fortran-int":
            data, _ = make_npy(version, fortran=1)
        stream = CheckedStream(data)
        stream.seek(8)
        expect_header_error(stream, version)
        if variant in ("empty-length", "oversized-length"):
            assert stream.requests == [
                (8, size)
            ], "length must be rejected before any header body read"
        return
    if kind == "object":
        settings: Dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
        payload = b"N."
        if variant == "unsafe":
            payload = b"cos\nsystem\n."
        elif variant == "partial":
            payload = b"cos\nsystem\n\x00"
        elif variant == "argument-cap":
            settings.setdefault("scan", {})["max_pickle_argument_bytes"] = 256
            payload = b"\x80\x04\x8d" + struct.pack("<Q", 257)
        elif variant == "opcode-cap":
            settings.setdefault("scan", {})["max_pickle_opcodes"] = 1
            payload = b"NN."
        elif variant == "memo-cap":
            settings.setdefault("scan", {})["max_pickle_memo_entries"] = 1
            payload = b"\x80\x04N\x94N\x94."
        elif variant == "byte-cap":
            settings.setdefault("scan", {})["max_pickle_bytes"] = 256
            payload = b"N" * 300 + b"."
        original_settings = settings_snapshot(settings)
        data, end = make_npy(version, descr="|O", payload=payload)
        model = OffsetModel(CheckedStream(data))
        result = scan_numpy(model, settings)
        assert (
            end in model.offsets
        ), "pickle disassembly must begin at the original payload offset"
        assert settings_snapshot(settings) == original_settings
        assert bool(result.issues) == (variant in ("unsafe", "partial"))
        assert bool(result.errors) == (variant not in ("safe", "unsafe"))
        assert not result.skipped
        if variant in ("unsafe", "partial"):
            assert any(
                isinstance(issue.details, OperatorIssueDetails)
                and issue.details.module == "os"
                and issue.details.operator == "system"
                for issue in result.issues
            )
        return
    if kind == "v3-invalid":
        base = "{'descr': '<i4', 'fortran_order': False, 'shape': (1,)}"
        invalid = {
            "duplicate": "{'descr': '|O', 'descr': '<i4', 'fortran_order': False, 'shape': (1,)}",
            "extra": base[:-1] + ", 'unexpected': 1}",
            "missing": "{'descr': '<i4', 'shape': (1,)}",
            "not-dict": "[]",
            "shape-list": base.replace("(1,)", "[1]"),
            "dtype-none": base.replace("'<i4'", "None"),
            "dtype-dict": base.replace("'<i4'", "{}"),
            "bad-dtype": base.replace("'<i4'", "'not-a-dtype'"),
            "expression": base.replace("'<i4'", "open('MUST_NOT_EXECUTE')"),
            "invalid-utf8": b"{'descr': [('\xff', '<i4')], 'fortran_order': False, 'shape': (1,)}",
            "deep-recursion": base.replace("'<i4'", "(" * 500 + "'i4'" + ")" * 500),
        }[variant]
        data, _ = make_npy(version, text=invalid)
        stream = CheckedStream(data)
        stream.seek(8)
        expect_header_error(stream, version)
        assert not Path("MUST_NOT_EXECUTE").exists()
        return
    if kind == "unsupported":
        stream = CheckedStream(b"not read")
        expect_header_error(stream, version)
        assert stream.requests == []
        return
    if kind == "header-limit":
        data, _ = make_npy((3, 0))
        settings = {
            "scan": {
                (
                    "max_pickle_argument_bytes"
                    if variant != "bytes"
                    else "max_pickle_bytes"
                ): 64
            }
        }
        if variant == "invalid-setting":
            settings["scan"]["max_pickle_argument_bytes"] = True
        stream = CheckedStream(data)
        stream.seek(8)
        expect_header_error(stream, (3, 0), settings)
        assert all(size <= 4 for _, size in stream.requests)
        return
    if variant == "nonseek":

        class NonSeek(io.BytesIO):
            def seek(self, *args: Any, **kwargs: Any) -> int:
                raise io.UnsupportedOperation("no seeking")

        data, _ = make_npy((3, 0))
        try:
            scan_numpy(Model(Path("nonseek.npy"), NonSeek(data)), DEFAULT_SETTINGS)
        except OSError:
            return
        raise AssertionError("nonseekable stream must not produce a clean scan")
    if variant in ("raw-safe", "raw-unsafe", "zip"):
        data = {
            "raw-safe": b"N.",
            "raw-unsafe": b"cos\nsystem\n.",
            "zip": b"PK\x03\x04",
        }[variant]
        result = scan_numpy(
            Model(Path("fallback.npy"), io.BytesIO(data)), DEFAULT_SETTINGS
        )
        assert bool(result.issues) == (variant == "raw-unsafe")
        assert bool(result.skipped) == (variant == "zip")
        assert not result.errors
        return
    if variant in (
        "integration-header",
        "integration-object",
        "truncated-magic",
        "truncated-version",
    ):
        data, _ = make_npy((3, 0), text="{'descr': '<i4', 'shape': (1,)}")
        if variant == "integration-object":
            data, _ = make_npy((3, 0), descr="|O", payload=b"cos\nsystem\n\x00")
        if variant == "truncated-magic":
            data = b"\x93NUM"
        elif variant == "truncated-version":
            data = b"\x93NUMPY\x03"
        with tempfile.TemporaryDirectory(prefix="npy-header-case-") as directory:
            path = Path(directory) / "fixture.npy"
            path.write_bytes(data)
            report = ModelScan().scan(path)
            assert (
                report["errors"] and report["summary"]["scanned"]["total_scanned"] == 0
            )
            if variant in ("integration-header", "integration-object"):
                assert bool(report["issues"]) == (variant == "integration-object")
        return
    raise AssertionError(f"unknown fixture: {case}")

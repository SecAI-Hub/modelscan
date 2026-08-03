import copy
import io
import pickle
import struct
import zipfile
from pathlib import Path
from typing import Any, Dict, cast

import pytest

from modelscan.model import Model, ModelFileChangedError
from modelscan.modelscan import ModelScan
from modelscan.scanners.keras.scan import KerasLambdaDetectScan
from modelscan.settings import DEFAULT_SETTINGS
from modelscan.tools.archive import ArchiveLimitError, safe_zip_members


def _replace_first_stored_member_byte(archive_path: Path) -> None:
    raw = bytearray(archive_path.read_bytes())
    local_header = raw.index(b"PK\x03\x04")
    name_length, extra_length = struct.unpack_from("<HH", raw, local_header + 26)
    data_offset = local_header + 30 + name_length + extra_length
    raw[data_offset] ^= 0x01
    archive_path.write_bytes(raw)


def _replace_zip_compression_method(archive_path: Path, method: int) -> None:
    raw = bytearray(archive_path.read_bytes())
    local_header = raw.index(b"PK\x03\x04")
    central_header = raw.index(b"PK\x01\x02")
    struct.pack_into("<H", raw, local_header + 8, method)
    struct.pack_into("<H", raw, central_header + 10, method)
    archive_path.write_bytes(raw)


def test_model_open_rejects_symbolic_links(tmp_path: Path) -> None:
    target = tmp_path / "target.pkl"
    target.write_bytes(b"safe test content")
    link = tmp_path / "linked.pkl"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(OSError, match="regular file"):
        with Model(link):
            pass


def test_model_open_rejects_hard_links(tmp_path: Path) -> None:
    target = tmp_path / "target.pkl"
    target.write_bytes(b"safe test content")
    link = tmp_path / "linked.pkl"
    try:
        link.hardlink_to(target)
    except OSError:
        pytest.skip("hard links are unavailable on this platform")

    with pytest.raises(OSError, match="single-link"):
        with Model(link):
            pass


def test_model_detects_in_place_change_before_context_exit(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    path.write_bytes(b"initial model")

    with pytest.raises(ModelFileChangedError, match="changed while"):
        with Model(path):
            path.write_bytes(b"changed model")


def test_directory_scan_rejects_links_in_untrusted_tree(tmp_path: Path) -> None:
    target = tmp_path / "outside.pkl"
    target.write_bytes(b"outside")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    try:
        (model_dir / "linked.pkl").symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    report = ModelScan().scan(model_dir)

    assert report["summary"]["scanned"]["total_scanned"] == 0
    assert report["errors"][0]["category"] == "PATH"
    assert "symbolic links" in report["errors"][0]["description"]


def test_directory_file_count_is_bounded(tmp_path: Path) -> None:
    (tmp_path / "one.pkl").write_bytes(b"one")
    (tmp_path / "two.pkl").write_bytes(b"two")
    settings: Dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
    settings["scan"]["max_files"] = 1

    report = ModelScan(settings).scan(tmp_path)

    assert report["summary"]["scanned"]["total_scanned"] == 0
    assert "file limit" in report["errors"][0]["description"]


def test_directory_entry_count_is_bounded(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    settings: Dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
    settings["scan"]["max_entries"] = 1

    report = ModelScan(settings).scan(tmp_path)

    assert report["summary"]["scanned"]["total_scanned"] == 0
    assert "entry limit" in report["errors"][0]["description"]


def test_directory_scan_discards_results_when_a_file_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.pkl"
    model.write_bytes(pickle.dumps({"safe": True}))
    scanner = ModelScan()
    original_scan_source = scanner._scan_source

    def scan_then_change(candidate: Model) -> bool:
        result = original_scan_source(candidate)
        if candidate.get_source() == model:
            model.write_bytes(model.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(scanner, "_scan_source", scan_then_change)

    report = scanner.scan(tmp_path)

    assert report["summary"]["scanned"]["total_scanned"] == 0
    assert report["issues"] == []
    assert any(
        error["category"] == "PATH" and "changed" in error["description"]
        for error in report["errors"]
    )


def test_directory_scan_detects_entries_added_during_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.pkl"
    model.write_bytes(pickle.dumps({"safe": True}))
    scanner = ModelScan()
    original_scan_source = scanner._scan_source
    added = False

    def scan_then_add(candidate: Model) -> bool:
        nonlocal added
        result = original_scan_source(candidate)
        if not added:
            (tmp_path / "late.pkl").write_bytes(pickle.dumps({"late": True}))
            added = True
        return result

    monkeypatch.setattr(scanner, "_scan_source", scan_then_add)

    report = scanner.scan(tmp_path)

    assert report["summary"]["scanned"]["total_scanned"] == 0
    assert report["summary"]["skipped"]["total_skipped"] == 0
    assert any(
        error["category"] == "PATH" and "tree changed" in error["description"]
        for error in report["errors"]
    )


def test_direct_file_size_is_bounded(tmp_path: Path) -> None:
    model = tmp_path / "large.pkl"
    model.write_bytes(b"12345")
    settings: Dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
    settings["scan"]["max_file_size"] = 4

    report = ModelScan(settings).scan(model)

    assert report["summary"]["scanned"]["total_scanned"] == 0
    assert "size limit" in report["errors"][0]["description"]


def test_modelscan_instances_do_not_share_mutated_settings() -> None:
    first = ModelScan()
    first._settings["archive"]["max_members"] = 1

    second = ModelScan()

    default_archive = cast(Dict[str, Any], DEFAULT_SETTINGS["archive"])
    assert default_archive["max_members"] == 10000
    assert second._settings["archive"]["max_members"] == 10000


def test_archive_rejects_duplicate_member_names(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("model.pkl", b"one")
        archive.writestr("model.pkl", b"two")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ArchiveLimitError, match="duplicate"):
            safe_zip_members(archive, DEFAULT_SETTINGS, str(archive_path))


def test_archive_rejects_excessive_compression_ratio(tmp_path: Path) -> None:
    archive_path = tmp_path / "compressed.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("model.pkl", b"A" * 100_000)
    settings: Dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
    settings["archive"]["max_compression_ratio"] = 2

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ArchiveLimitError, match="compression-ratio"):
            safe_zip_members(archive, settings, str(archive_path))


def test_archive_settings_reject_nonpositive_limits(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("model.pkl", b"data")
    settings: Dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
    settings["archive"]["max_members"] = 0

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ArchiveLimitError, match="positive integer"):
            safe_zip_members(archive, settings, str(archive_path))


def test_archive_rejects_unsupported_compression_method(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsupported.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("model.pkl", b"data")
    _replace_zip_compression_method(archive_path, 99)

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ArchiveLimitError, match="unsupported"):
            safe_zip_members(archive, DEFAULT_SETTINGS, str(archive_path))


def test_corrupt_zip_member_is_reported_as_bad_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr("model.pkl", pickle.dumps({"safe": True}))
    _replace_first_stored_member_byte(archive_path)

    report = ModelScan().scan(archive_path)

    assert report["summary"]["scanned"]["total_scanned"] == 0
    assert report["issues"] == []
    skipped = report["summary"]["skipped"]["skipped_files"]
    bad_zip = [entry for entry in skipped if entry["category"] == "BAD_ZIP"]
    assert len(bad_zip) == 1
    assert "safely decompressed" in bad_zip[0]["description"]


def test_keras_config_read_is_bounded() -> None:
    settings: Dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
    settings["archive"]["max_config_json_size"] = 4
    scanner = KerasLambdaDetectScan(settings)
    model = Model("archive.keras:config.json", io.BytesIO(b'{"config": {}}'))

    with pytest.raises(ValueError, match="size limit"):
        scanner._get_keras_operator_names(model)

import zipfile
from contextlib import contextmanager
from tempfile import SpooledTemporaryFile
from typing import IO, Any, Dict, Iterator, List

DEFAULT_MAX_ZIP_MEMBERS = 10000
DEFAULT_MAX_ZIP_MEMBER_SIZE = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_ZIP_TOTAL_UNCOMPRESSED_SIZE = 10 * 1024 * 1024 * 1024
DEFAULT_MAX_ZIP_MEMBER_NAME_BYTES = 4096
DEFAULT_MAX_ZIP_COMPRESSION_RATIO = 100000
DEFAULT_MAX_CONFIG_JSON_SIZE = 16 * 1024 * 1024
DEFAULT_ZIP_SPOOL_MEMORY = 8 * 1024 * 1024


class ArchiveLimitError(Exception):
    pass


def _supported_compression_methods() -> set[int]:
    """Return compression methods this interpreter can safely decompress."""
    methods = {zipfile.ZIP_STORED}
    for constant, module in (
        (zipfile.ZIP_DEFLATED, "zlib"),
        (zipfile.ZIP_BZIP2, "bz2"),
        (zipfile.ZIP_LZMA, "lzma"),
    ):
        if getattr(zipfile, module, None) is not None:
            methods.add(constant)
    return methods


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ArchiveLimitError(f"archive setting {name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ArchiveLimitError(
            f"archive setting {name} must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise ArchiveLimitError(f"archive setting {name} must be a positive integer")
    return parsed


def _archive_settings(settings: Dict[str, Any]) -> Dict[str, int]:
    configured = settings.get("archive", {})
    if not isinstance(configured, dict):
        raise ArchiveLimitError("archive settings must be a mapping")
    return {
        "max_members": _positive_int(
            configured.get("max_members", DEFAULT_MAX_ZIP_MEMBERS), "max_members"
        ),
        "max_member_size": _positive_int(
            configured.get("max_member_size", DEFAULT_MAX_ZIP_MEMBER_SIZE),
            "max_member_size",
        ),
        "max_total_uncompressed_size": _positive_int(
            configured.get(
                "max_total_uncompressed_size",
                DEFAULT_MAX_ZIP_TOTAL_UNCOMPRESSED_SIZE,
            ),
            "max_total_uncompressed_size",
        ),
        "max_member_name_bytes": _positive_int(
            configured.get("max_member_name_bytes", DEFAULT_MAX_ZIP_MEMBER_NAME_BYTES),
            "max_member_name_bytes",
        ),
        "max_compression_ratio": _positive_int(
            configured.get("max_compression_ratio", DEFAULT_MAX_ZIP_COMPRESSION_RATIO),
            "max_compression_ratio",
        ),
        "max_config_json_size": _positive_int(
            configured.get("max_config_json_size", DEFAULT_MAX_CONFIG_JSON_SIZE),
            "max_config_json_size",
        ),
    }


def max_config_json_size(settings: Dict[str, Any]) -> int:
    """Return the validated bound for archive-contained JSON configuration."""
    return _archive_settings(settings)["max_config_json_size"]


def safe_zip_members(
    archive: zipfile.ZipFile,
    settings: Dict[str, Any],
    source: str,
) -> List[zipfile.ZipInfo]:
    limits = _archive_settings(settings)
    all_members = archive.infolist()

    if len(all_members) > limits["max_members"]:
        raise ArchiveLimitError(
            f"{source} has {len(all_members)} zip entries, exceeding "
            f"the configured limit of {limits['max_members']}."
        )
    members = [member for member in all_members if not member.is_dir()]

    names: set[str] = set()
    supported_compression = _supported_compression_methods()
    for member in all_members:
        if member.filename in names:
            raise ArchiveLimitError(
                f"{source} contains duplicate zip entry {member.filename!r}."
            )
        names.add(member.filename)
        if len(member.filename.encode("utf-8")) > limits["max_member_name_bytes"]:
            raise ArchiveLimitError(f"{source} contains an overlong zip entry name.")
        if member.flag_bits & 0x1:
            raise ArchiveLimitError(
                f"File {member.filename!r} is encrypted, password required for extraction"
            )
        if member.compress_type not in supported_compression:
            raise ArchiveLimitError(
                f"{source}:{member.filename} uses unsupported zip compression "
                f"method {member.compress_type}."
            )

    oversized_member = next(
        (member for member in members if member.file_size > limits["max_member_size"]),
        None,
    )
    if oversized_member:
        raise ArchiveLimitError(
            f"{source}:{oversized_member.filename} declares "
            f"{oversized_member.file_size} uncompressed bytes, exceeding "
            f"the configured per-entry limit of {limits['max_member_size']}."
        )

    suspicious_ratio = next(
        (
            member
            for member in members
            if member.file_size
            and member.file_size
            > max(member.compress_size, 1) * limits["max_compression_ratio"]
        ),
        None,
    )
    if suspicious_ratio:
        raise ArchiveLimitError(
            f"{source}:{suspicious_ratio.filename} exceeds the configured "
            "compression-ratio limit."
        )

    total_uncompressed_size = sum(member.file_size for member in members)
    if total_uncompressed_size > limits["max_total_uncompressed_size"]:
        raise ArchiveLimitError(
            f"{source} declares {total_uncompressed_size} total uncompressed "
            f"bytes, exceeding the configured limit of "
            f"{limits['max_total_uncompressed_size']}."
        )

    return members


@contextmanager
def verified_zip_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    settings: Dict[str, Any],
    source: str,
) -> Iterator[IO[bytes]]:
    """Materialize one bounded member and verify decompression, length, and CRC."""
    limits = _archive_settings(settings)
    spool_limit = min(DEFAULT_ZIP_SPOOL_MEMORY, limits["max_member_size"])
    temporary = SpooledTemporaryFile(max_size=spool_limit, mode="w+b")
    total = 0
    try:
        try:
            with archive.open(member, "r") as member_stream:
                while True:
                    chunk = member_stream.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > member.file_size or total > limits["max_member_size"]:
                        raise ArchiveLimitError(
                            f"{source}:{member.filename} expands beyond its declared "
                            "or configured size."
                        )
                    temporary.write(chunk)
        except ArchiveLimitError:
            raise
        except Exception as exc:
            raise ArchiveLimitError(
                f"{source}:{member.filename} could not be safely decompressed "
                f"({type(exc).__name__})."
            ) from exc

        if total != member.file_size:
            raise ArchiveLimitError(
                f"{source}:{member.filename} decompressed to {total} bytes, not "
                f"the declared {member.file_size} bytes."
            )

        temporary.seek(0)
        yield temporary
    finally:
        temporary.close()

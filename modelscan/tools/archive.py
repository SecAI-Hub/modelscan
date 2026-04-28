import zipfile
from typing import Any, Dict, List

DEFAULT_MAX_ZIP_MEMBERS = 10000
DEFAULT_MAX_ZIP_MEMBER_SIZE = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_ZIP_TOTAL_UNCOMPRESSED_SIZE = 10 * 1024 * 1024 * 1024


class ArchiveLimitError(Exception):
    pass


def _archive_settings(settings: Dict[str, Any]) -> Dict[str, int]:
    configured = settings.get("archive", {})
    return {
        "max_members": int(configured.get("max_members", DEFAULT_MAX_ZIP_MEMBERS)),
        "max_member_size": int(
            configured.get("max_member_size", DEFAULT_MAX_ZIP_MEMBER_SIZE)
        ),
        "max_total_uncompressed_size": int(
            configured.get(
                "max_total_uncompressed_size",
                DEFAULT_MAX_ZIP_TOTAL_UNCOMPRESSED_SIZE,
            )
        ),
    }


def safe_zip_members(
    archive: zipfile.ZipFile,
    settings: Dict[str, Any],
    source: str,
) -> List[zipfile.ZipInfo]:
    limits = _archive_settings(settings)
    members = [member for member in archive.infolist() if not member.is_dir()]

    if len(members) > limits["max_members"]:
        raise ArchiveLimitError(
            f"{source} has {len(members)} zip entries, exceeding "
            f"the configured limit of {limits['max_members']}."
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

    total_uncompressed_size = sum(member.file_size for member in members)
    if total_uncompressed_size > limits["max_total_uncompressed_size"]:
        raise ArchiveLimitError(
            f"{source} declares {total_uncompressed_size} total uncompressed "
            f"bytes, exceeding the configured limit of "
            f"{limits['max_total_uncompressed_size']}."
        )

    return members

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Dict, Optional, Union


class ModelDataEmpty(ValueError):
    pass


class ModelFileChangedError(OSError):
    """Raised when a path no longer matches the file opened for scanning."""


@dataclass(frozen=True)
class FileIdentity:
    """Security-relevant metadata used to detect intake path replacement."""

    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> "FileIdentity":
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            links=metadata.st_nlink,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
            changed_ns=metadata.st_ctime_ns,
        )


class Model:
    _source: Path
    _stream: Optional[IO[bytes]]
    _should_close_stream: bool  # Flag to control closing of file
    _context: Dict[str, Any]

    def __init__(
        self,
        source: Union[str, Path],
        stream: Optional[IO[bytes]] = None,
        *,
        expected_identity: Optional[FileIdentity] = None,
        integrity_verifier: Optional[Callable[[], None]] = None,
        integrity_source: Optional[Union[str, Path]] = None,
    ):
        self._source = Path(source)
        self._stream = stream
        self._should_close_stream = stream is None  # Only close if opened
        self._context = {"formats": []}
        self._expected_identity = expected_identity
        self._opened_identity: Optional[FileIdentity] = None
        self._integrity_verifier = integrity_verifier
        self._integrity_source = Path(integrity_source or source)

    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = value

    def get_context(self, key: str) -> Any:
        return self._context.get(key)

    def open(self) -> "Model":
        if self._stream:
            return self

        before = self._source.lstat()
        before_identity = FileIdentity.from_stat(before)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise OSError("model path must be a single-link regular file")
        if (
            self._expected_identity is not None
            and before_identity != self._expected_identity
        ):
            raise ModelFileChangedError(
                "model path changed after it was selected for scanning"
            )
        descriptor = os.open(
            self._source,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        try:
            opened = os.fstat(descriptor)
            opened_identity = FileIdentity.from_stat(opened)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened_identity != before_identity
            ):
                raise ModelFileChangedError(
                    "model path changed while it was being opened"
                )
            self._stream = os.fdopen(descriptor, "rb")
            self._opened_identity = opened_identity
        except Exception:
            os.close(descriptor)
            raise
        self._should_close_stream = True

        return self

    def close(self) -> None:
        # Only close the stream if we opened a file (not for IO[bytes] objects passed in)
        if self._stream and self._should_close_stream:
            self._stream.close()
            self._stream = None  # Avoid double-closing
            self._should_close_stream = False  # Reset the flag

    def __enter__(self) -> "Model":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        try:
            if exc_type is None:
                self.verify_unchanged()
        finally:
            self.close()

    def get_source(self) -> Path:
        return self._source

    def get_integrity_source(self) -> Path:
        return self._integrity_source

    def verify_unchanged(self) -> None:
        """Verify that scanners read the same regular file selected at intake."""
        if self._integrity_verifier is not None:
            self._integrity_verifier()
            return

        if self._opened_identity is None or self._stream is None:
            return

        try:
            descriptor_identity = FileIdentity.from_stat(
                os.fstat(self._stream.fileno())
            )
            path_metadata = self._source.lstat()
            path_identity = FileIdentity.from_stat(path_metadata)
        except (OSError, ValueError) as exc:
            raise ModelFileChangedError(
                "model path could not be verified after scanning"
            ) from exc

        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or descriptor_identity != self._opened_identity
            or path_identity != self._opened_identity
        ):
            raise ModelFileChangedError("model path changed while it was being scanned")

    def get_stream(self, offset: int = 0) -> IO[bytes]:
        if not self._stream:
            raise ModelDataEmpty("Model data is empty.")

        self._stream.seek(offset)
        return self._stream

import copy
import importlib
import logging
import os
import stat
from dataclasses import dataclass

from modelscan.settings import DEFAULT_SETTINGS

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union
from datetime import datetime
import zipfile

from modelscan.error import (
    ModelScanError,
    PathError,
    ErrorBase,
    ModelScanScannerError,
    NestedZipError,
)
from modelscan.skip import ModelScanSkipped, SkipCategories
from modelscan.issues import Issues, IssueSeverity
from modelscan.scanners.scan import ScanBase
from modelscan._version import __version__
from modelscan.tools.archive import safe_zip_members, verified_zip_member
from modelscan.tools.utils import _is_zipfile
from modelscan.model import FileIdentity, Model, ModelFileChangedError
from modelscan.middlewares.middleware import MiddlewarePipeline, MiddlewareImportError

logger = logging.getLogger("modelscan")


@dataclass(frozen=True)
class _FileCandidate:
    path: Path
    identity: FileIdentity


@dataclass(frozen=True)
class _DirectorySnapshot:
    files: tuple[_FileCandidate, ...]
    entries: tuple[tuple[str, str, FileIdentity], ...]


class ModelScan:
    def __init__(
        self,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Output
        self._issues = Issues()
        self._errors: List[ErrorBase] = []
        self._init_errors: List[ModelScanError] = []
        self._skipped: List[ModelScanSkipped] = []
        self._scanned: List[str] = []
        self._input_path: str = ""

        # Scanners
        self._scanners_to_run: List[ScanBase] = []
        # Scanner and CLI code annotate settings at runtime. Keep each scan
        # instance isolated instead of mutating DEFAULT_SETTINGS or its caller.
        self._settings: Dict[str, Any] = copy.deepcopy(
            DEFAULT_SETTINGS if settings is None else settings
        )
        self._load_scanners()
        self._load_middlewares()

    def _scan_limits(self) -> Dict[str, int]:
        configured = self._settings.get("scan", {})
        if not isinstance(configured, dict):
            raise ValueError("scan limits must be a mapping")
        defaults = {
            "max_files": 100000,
            "max_entries": 100000,
            "max_depth": 64,
            "max_path_bytes": 4096,
            "max_file_size": 2 * 1024**4,
            "max_total_size": 10 * 1024**4,
        }
        limits: Dict[str, int] = {}
        for name, default in defaults.items():
            value = configured.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"scan limit {name} must be a positive integer")
            limits[name] = value
        return limits

    def _snapshot_directory(self, root: Path) -> _DirectorySnapshot:
        """Build a bounded, no-follow identity manifest for an untrusted tree."""
        limits = self._scan_limits()

        root_metadata = root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise ValueError("model tree root must be a non-symbolic-link directory")

        files: List[_FileCandidate] = []
        manifest: List[tuple[str, str, FileIdentity]] = [
            (".", "directory", FileIdentity.from_stat(root_metadata))
        ]
        entry_count = 0
        total_size = 0
        pending = [(root, 0)]
        while pending:
            directory, parent_depth = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    entry_count += 1
                    if entry_count > limits["max_entries"]:
                        raise ValueError(
                            "model tree exceeds the configured entry limit"
                        )
                    path = Path(entry.path)
                    relative = path.relative_to(root)
                    if (
                        len(relative.as_posix().encode("utf-8"))
                        > limits["max_path_bytes"]
                    ):
                        raise ValueError("model tree contains an overlong path")
                    metadata = entry.stat(follow_symlinks=False)
                    identity = FileIdentity.from_stat(metadata)
                    if stat.S_ISLNK(metadata.st_mode):
                        raise ValueError(f"symbolic links are not scanned: {relative}")
                    if stat.S_ISDIR(metadata.st_mode):
                        depth = parent_depth + 1
                        if depth > limits["max_depth"]:
                            raise ValueError(
                                "model tree exceeds the configured depth limit"
                            )
                        manifest.append((relative.as_posix(), "directory", identity))
                        pending.append((path, depth))
                    elif stat.S_ISREG(metadata.st_mode):
                        if metadata.st_nlink != 1:
                            raise ValueError("hard-linked model files are not scanned")
                        if metadata.st_size > limits["max_file_size"]:
                            raise ValueError(
                                "model tree contains a file exceeding the configured size limit"
                            )
                        total_size += metadata.st_size
                        if total_size > limits["max_total_size"]:
                            raise ValueError(
                                "model tree exceeds the configured total size limit"
                            )
                        manifest.append((relative.as_posix(), "file", identity))
                        files.append(_FileCandidate(path, identity))
                        if len(files) > limits["max_files"]:
                            raise ValueError(
                                "model tree exceeds the configured file limit"
                            )
                    else:
                        raise ValueError(f"special files are not scanned: {relative}")
        return _DirectorySnapshot(
            files=tuple(
                sorted(
                    files,
                    key=lambda candidate: candidate.path.relative_to(root).as_posix(),
                )
            ),
            entries=tuple(sorted(manifest, key=lambda entry: (entry[0], entry[1]))),
        )

    def _stable_directory_snapshot(self, root: Path) -> _DirectorySnapshot:
        first = self._snapshot_directory(root)
        second = self._snapshot_directory(root)
        if first.entries != second.entries:
            raise ValueError("model tree changed while it was being enumerated")
        return second

    def _directory_files(self, root: Path) -> List[Path]:
        """Return paths from a stable no-follow directory snapshot."""
        return [
            candidate.path for candidate in self._stable_directory_snapshot(root).files
        ]

    @staticmethod
    def _source_matches_file(source: Union[str, Path], file: Path) -> bool:
        container, _ = ModelScan._split_archive_source(source)
        return Path(container) == file

    def _discard_results_for_file(
        self,
        file: Path,
        *,
        discard_skips: bool = True,
    ) -> None:
        """Discard any apparent success produced from a changed/corrupt file."""
        self._scanned = [
            source
            for source in self._scanned
            if not self._source_matches_file(source, file)
        ]
        if discard_skips:
            self._skipped = [
                skipped
                for skipped in self._skipped
                if not self._source_matches_file(skipped.source, file)
            ]
        self._issues.all_issues = [
            issue
            for issue in self._issues.all_issues
            if not self._source_matches_file(
                getattr(issue.details, "source", ""),
                file,
            )
        ]

    def _record_integrity_error(self, file: Path, error: Exception) -> None:
        self._discard_results_for_file(file)
        message = f"model changed during security scan: {error}"
        if not any(
            isinstance(existing, PathError)
            and existing.path == file
            and existing.message == message
            for existing in self._errors
        ):
            self._errors.append(PathError(message, file))

    def _load_middlewares(self) -> None:
        try:
            self._middleware_pipeline = MiddlewarePipeline.from_settings(
                self._settings["middlewares"] or {}
            )
        except MiddlewareImportError as e:
            logger.exception(e)
            self._init_errors.append(ModelScanError(f"Error loading middlewares: {e}"))

    def _load_scanners(self) -> None:
        for scanner_path, scanner_settings in self._settings["scanners"].items():
            if (
                "enabled" in scanner_settings.keys()
                and self._settings["scanners"][scanner_path]["enabled"]
            ):
                try:
                    modulename, classname = scanner_path.rsplit(".", 1)
                    imported_module = importlib.import_module(
                        name=modulename, package=classname
                    )

                    scanner_class: ScanBase = getattr(imported_module, classname)
                    self._scanners_to_run.append(scanner_class)

                except Exception as e:
                    logger.error("Error importing scanner %s", scanner_path)
                    self._init_errors.append(
                        ModelScanError(
                            f"Error importing scanner {scanner_path}: {e}",
                        )
                    )

    def _iterate_models(self, model_path: Path) -> Generator[Model, None, None]:
        directory_snapshot: Optional[_DirectorySnapshot] = None
        try:
            metadata = model_path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("symbolic-link model paths are not scanned")
            candidates = [_FileCandidate(model_path, FileIdentity.from_stat(metadata))]
            if stat.S_ISDIR(metadata.st_mode):
                logger.debug("Path %s is a directory", str(model_path))
                directory_snapshot = self._stable_directory_snapshot(model_path)
                candidates = list(directory_snapshot.files)
            elif not stat.S_ISREG(metadata.st_mode):
                raise ValueError("model path must be a regular file or directory")
            else:
                limits = self._scan_limits()
                if metadata.st_size > min(
                    limits["max_file_size"], limits["max_total_size"]
                ):
                    raise ValueError("model file exceeds the configured size limit")
                if metadata.st_nlink != 1:
                    raise ValueError("hard-linked model files are not scanned")
        except FileNotFoundError:
            logger.error("Path %s does not exist", model_path)
            self._errors.append(PathError("Path is not valid", model_path))
            return
        except (OSError, UnicodeError, ValueError) as exc:
            logger.error("Unsafe model path %s: %s", model_path, exc)
            self._errors.append(PathError(str(exc), model_path))
            return

        for candidate in candidates:
            file = candidate.path
            try:
                with Model(file, expected_identity=candidate.identity) as model:
                    yield model
                    if model.get_context("_modelscan_integrity_error"):
                        raise ModelFileChangedError(
                            "model identity check failed after scanner execution"
                        )
                    model.verify_unchanged()

                    if not _is_zipfile(file, model.get_stream()):
                        continue

                    try:
                        with zipfile.ZipFile(model.get_stream(), "r") as archive:
                            members = safe_zip_members(
                                archive,
                                self._settings,
                                str(model.get_source()),
                            )
                            for member in members:
                                with verified_zip_member(
                                    archive,
                                    member,
                                    self._settings,
                                    str(model.get_source()),
                                ) as file_io:
                                    model.verify_unchanged()
                                    file_name = (
                                        f"{model.get_source()}:{member.filename}"
                                    )
                                    if _is_zipfile(file_name, data=file_io):
                                        self._errors.append(
                                            NestedZipError(
                                                "ModelScan does not support nested zip files.",
                                                Path(file_name),
                                            )
                                        )
                                        continue

                                    archived_model = Model(
                                        file_name,
                                        file_io,
                                        integrity_verifier=model.verify_unchanged,
                                        integrity_source=model.get_source(),
                                    )
                                    yield archived_model
                                    if archived_model.get_context(
                                        "_modelscan_integrity_error"
                                    ):
                                        raise ModelFileChangedError(
                                            "archive changed during member scanning"
                                        )
                                    archived_model.verify_unchanged()
                    except ModelFileChangedError:
                        raise
                    except Exception as e:
                        logger.debug(
                            "Skipping zip file %s, due to error",
                            str(model.get_source()),
                            exc_info=True,
                        )
                        self._discard_results_for_file(file, discard_skips=False)
                        self._skipped.append(
                            ModelScanSkipped(
                                "ModelScan",
                                SkipCategories.BAD_ZIP,
                                f"Skipping zip file due to error: {e}",
                                str(model.get_source()),
                            )
                        )
            except ModelFileChangedError as exc:
                logger.error("Model path changed during scan %s: %s", file, exc)
                self._record_integrity_error(file, exc)
            except (OSError, ValueError) as exc:
                logger.error("Unable to securely open model path %s: %s", file, exc)
                self._errors.append(PathError(str(exc), file))

        if directory_snapshot is not None:
            try:
                final_snapshot = self._stable_directory_snapshot(model_path)
                if final_snapshot.entries != directory_snapshot.entries:
                    raise ModelFileChangedError(
                        "model tree changed while it was being scanned"
                    )
            except (OSError, UnicodeError, ValueError) as exc:
                self._issues = Issues()
                self._scanned = []
                self._skipped = []
                self._record_integrity_error(model_path, exc)

    def scan(
        self,
        path: Union[str, Path],
    ) -> Dict[str, Any]:
        self._issues = Issues()
        self._errors = []
        self._errors.extend(self._init_errors)
        self._skipped = []
        self._scanned = []
        self._input_path = str(path)
        pathlib_path = Path().cwd() if path == "." else Path(path).absolute()
        model_path = Path(pathlib_path)

        all_paths: List[Path] = []
        for model in self._iterate_models(model_path):
            self._middleware_pipeline.run(model)
            self._scan_source(model)
            try:
                model.verify_unchanged()
            except ModelFileChangedError as exc:
                self._record_integrity_error(model.get_integrity_source(), exc)
                model.set_context("_modelscan_integrity_error", True)
            all_paths.append(model.get_source())

        if self._skipped:
            all_skipped_paths = [skipped.source for skipped in self._skipped]
            for path in all_paths:
                main_file_path, _ = self._split_archive_source(path)

                if main_file_path == str(path):
                    continue

                # If main container is skipped, we only add its content to skipped but not the file itself
                if main_file_path in all_skipped_paths:
                    self._skipped = [
                        item for item in self._skipped if item.source != main_file_path
                    ]

                    continue

        return self._generate_results()

    @staticmethod
    def _split_archive_source(source: Union[str, Path]) -> tuple[str, str]:
        source_str = str(source)
        drive, path_without_drive = os.path.splitdrive(source_str)
        archive_separator_index = path_without_drive.find(":")

        if archive_separator_index >= 0:
            return (
                drive + path_without_drive[:archive_separator_index],
                path_without_drive[archive_separator_index + 1 :],
            )

        return source_str, ""

    @staticmethod
    def _relative_source_path(source: Union[str, Path], base_path: Path) -> str:
        source_str = str(source)
        source_path, archive_member = ModelScan._split_archive_source(source)

        try:
            relative_path = str(Path(source_path).relative_to(base_path))
        except ValueError:
            relative_path = source_str

        if archive_member and relative_path != source_str:
            return f"{relative_path}:{archive_member}"
        return relative_path

    def _scan_source(
        self,
        model: Model,
    ) -> bool:
        scanned = False
        for scan_class in self._scanners_to_run:
            scanner = scan_class(self._settings)  # type: ignore[operator]

            try:
                scan_results = scanner.scan(model)
            except Exception as e:
                logger.error(
                    "Error encountered from scanner %s with path %s: %s",
                    scanner.full_name(),
                    str(model.get_source()),
                    e,
                )
                self._errors.append(
                    ModelScanScannerError(
                        scanner.full_name(),
                        str(e),
                        model,
                    )
                )
                continue

            if scan_results is not None:
                scanned = True
                logger.info(
                    "Scanning %s using %s model scan",
                    model.get_source(),
                    scanner.full_name(),
                )
                # Partial scans can carry both known findings and errors.
                # Preserve both, and count completion only with full coverage.
                self._errors.extend(scan_results.errors)
                self._issues.add_issues(scan_results.issues)
                self._skipped.extend(scan_results.skipped)
                if not scan_results.errors and not scan_results.skipped:
                    self._scanned.append(str(model.get_source()))

        if not scanned:
            all_skipped_files = [skipped.source for skipped in self._skipped]
            if str(model.get_source()) not in all_skipped_files:
                self._skipped.append(
                    ModelScanSkipped(
                        "ModelScan",
                        SkipCategories.SCAN_NOT_SUPPORTED,
                        "Model Scan did not scan file",
                        str(model.get_source()),
                    )
                )

        return scanned

    def _generate_results(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {}

        input_path = Path(self._input_path)
        absolute_path = input_path.absolute()
        if input_path.is_file() or (not input_path.exists() and input_path.suffix):
            absolute_path = Path(absolute_path).parent

        issues_by_severity = self._issues.group_by_severity()
        total_issue_count = len(self._issues.all_issues)

        report["summary"] = {"total_issues_by_severity": {}}
        for severity in IssueSeverity:
            if severity.name in issues_by_severity:
                report["summary"]["total_issues_by_severity"][severity.name] = len(
                    issues_by_severity[severity.name]
                )
            else:
                report["summary"]["total_issues_by_severity"][severity.name] = 0

        report["summary"]["total_issues"] = total_issue_count
        report["summary"]["input_path"] = str(self._input_path)
        report["summary"]["absolute_path"] = str(absolute_path)
        report["summary"]["modelscan_version"] = __version__
        report["summary"]["timestamp"] = datetime.now().isoformat()

        report["summary"]["scanned"] = {"total_scanned": len(self._scanned)}

        if self._scanned:
            scanned_files = []
            for file_name in self._scanned:
                scanned_files.append(
                    self._relative_source_path(file_name, absolute_path)
                )

            report["summary"]["scanned"]["scanned_files"] = scanned_files

        if self._issues.all_issues:
            report["issues"] = [
                issue.details.output_json() for issue in self._issues.all_issues
            ]

            for issue in report["issues"]:
                issue["source"] = self._relative_source_path(
                    issue["source"],
                    absolute_path,
                )
        else:
            report["issues"] = []

        all_errors = []
        if self._errors:
            for error in self._errors:
                error_information = error.to_dict()
                if "source" in error_information:
                    error_information["source"] = self._relative_source_path(
                        error_information["source"],
                        absolute_path,
                    )

                all_errors.append(error_information)

        report["errors"] = all_errors

        report["summary"]["skipped"] = {"total_skipped": len(self._skipped)}

        all_skipped_files = []
        if self._skipped:
            for skipped_file in self._skipped:
                skipped_file_information = {}
                skipped_file_information["category"] = str(skipped_file.category.name)
                skipped_file_information["description"] = str(skipped_file.message)
                skipped_file_information["source"] = self._relative_source_path(
                    skipped_file.source,
                    absolute_path,
                )
                all_skipped_files.append(skipped_file_information)

        report["summary"]["skipped"]["skipped_files"] = all_skipped_files

        return report

    def is_compatible(self, path: str) -> bool:
        # Determines whether a file path is compatible with any of the available scanners
        if Path(path).suffix in self._settings["supported_zip_extensions"]:
            return True
        for scanner_path, scanner_settings in self._settings["scanners"].items():
            if (
                "supported_extensions" in scanner_settings.keys()
                and Path(path).suffix
                in self._settings["scanners"][scanner_path]["supported_extensions"]
            ):
                return True

        return False

    def generate_report(self) -> Optional[str]:
        reporting_module = self._settings["reporting"]["module"]
        report_settings = self._settings["reporting"]["settings"]

        scan_report = None
        try:
            modulename, classname = reporting_module.rsplit(".", 1)
            imported_module = importlib.import_module(
                name=modulename, package=classname
            )

            report_class = getattr(imported_module, classname)
            scan_report = report_class.generate(scan=self, settings=report_settings)

        except Exception as e:
            logger.error("Error generating report using %s: %s", reporting_module, e)
            self._errors.append(
                ModelScanError(f"Error generating report using {reporting_module}: {e}")
            )

        return scan_report

    @property
    def issues(self) -> Issues:
        return self._issues

    @property
    def errors(self) -> List[ErrorBase]:
        return self._errors

    @property
    def scanned(self) -> List[str]:
        return self._scanned

    @property
    def skipped(self) -> List[ModelScanSkipped]:
        return self._skipped

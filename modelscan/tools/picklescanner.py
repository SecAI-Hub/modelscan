import ast
import io
import logging
import struct
import pickletools  # nosec
from tarfile import TarError
from typing import IO, Any, Dict, List, Set, Tuple, Union, Optional, cast

import numpy as np

from modelscan.error import PickleGenopsError
from modelscan.skip import ModelScanSkipped, SkipCategories
from modelscan.issues import Issue, IssueCode, IssueSeverity, OperatorIssueDetails
from modelscan.scanners.scan import ScanResults
from modelscan.model import Model

logger = logging.getLogger("modelscan")

from .utils import MAGIC_NUMBER, _should_read_directly, get_magic_number


class GenOpsError(Exception):
    def __init__(self, msg: str, globals: Optional[Set[Tuple[str, str]]]):
        self.msg = msg
        self.globals = globals
        super().__init__()

    def __str__(self) -> str:
        return self.msg


# TODO: handle methods loading other Pickle files (either mark as suspicious, or follow calls to scan other files [preventing infinite loops])
#
# pickle.loads()
# https://docs.python.org/3/library/pickle.html#pickle.loads
# pickle.load()
# https://docs.python.org/3/library/pickle.html#pickle.load
# numpy.load()
# https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load
# numpy.ctypeslib.load_library()
# https://numpy.org/doc/stable/reference/routines.ctypeslib.html#numpy.ctypeslib.load_library
# pandas.read_pickle()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html
# joblib.load()
# https://joblib.readthedocs.io/en/latest/generated/joblib.load.html
# torch.load()
# https://pytorch.org/docs/stable/generated/torch.load.html
# tf.keras.models.load_model()
# https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
#


class _BoundedPickleReader:
    """Bound lengths before pickletools can allocate an attacker-sized value."""

    def __init__(self, stream: IO[bytes], total: int, argument: int) -> None:
        self.stream = stream
        self.remaining = total
        self.argument = argument

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self.argument or size > self.remaining:
            raise ValueError("pickle byte or argument limit exceeded")
        value = self.stream.read(size)
        self.remaining -= len(value)
        return value

    def readline(self, size: int = -1) -> bytes:
        limit = min(self.argument, self.remaining)
        if size >= 0:
            limit = min(limit, size)
        value = self.stream.readline(limit + 1)
        if len(value) > limit:
            raise ValueError("pickle line or byte limit exceeded")
        self.remaining -= len(value)
        return value

    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.stream.seek(offset, whence)


def _pickle_limits(settings: Dict[str, Any]) -> Dict[str, int]:
    limits = {
        "max_pickle_bytes": 64 * 1024 * 1024,
        "max_pickle_argument_bytes": 1024 * 1024,
        "max_pickle_opcodes": 200000,
        "max_pickle_memo_entries": 100000,
    }
    configured = settings.get("scan", {})
    if not isinstance(configured, dict):
        raise ValueError("pickle scan limits must be a mapping")
    for key, maximum in limits.items():
        value = configured.get(key, maximum)
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError(f"{key} must be an integer from 1 to {maximum}")
        limits[key] = value
    return limits


def _list_globals(
    data: IO[bytes],
    multiple_pickles: bool = True,
    settings: Optional[Dict[str, Any]] = None,
) -> Set[Tuple[str, str]]:
    globals: Set[Any] = set()
    limits = _pickle_limits(settings or {})
    data = _BoundedPickleReader(data, limits["max_pickle_bytes"], limits["max_pickle_argument_bytes"])  # type: ignore[assignment]
    opcode_count = 0

    memo: Dict[Union[int, str], str] = {}
    # Scan the data for pickle buffers, stopping when parsing fails or stops making progress
    last_byte = b"dummy"
    while last_byte != b"":
        # Keep known operations even when a later opcode is malformed.
        parse_error = None
        try:
            ops: List[Tuple[Any, Any, Union[int, None]]] = []
            for op in pickletools.genops(data):
                opcode_count += 1
                if opcode_count > limits["max_pickle_opcodes"]:
                    raise ValueError("pickle opcode limit exceeded")
                ops.append(op)
        except Exception as e:
            parse_error = str(e)

        # Extract global imports
        for n in range(len(ops)):
            op = ops[n]
            op_name = op[0].name
            op_value: str = op[1]

            if (
                op_name in {"MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT"}
                and len(memo) >= limits["max_pickle_memo_entries"]
            ):
                raise GenOpsError("pickle memo limit exceeded", globals or None)
            if op_name == "MEMOIZE" and n > 0:
                memo[len(memo)] = ops[n - 1][1]
            elif op_name in ["PUT", "BINPUT", "LONG_BINPUT"] and n > 0:
                memo[op_value] = ops[n - 1][1]
            elif op_name in ("GLOBAL", "INST"):
                globals.add(tuple(op_value.split(" ", 1)))
            elif op_name == "STACK_GLOBAL":
                values: List[str] = []
                for offset in range(1, n):
                    if ops[n - offset][0].name in [
                        "MEMOIZE",
                        "PUT",
                        "BINPUT",
                        "LONG_BINPUT",
                    ]:
                        continue
                    if ops[n - offset][0].name in ["GET", "BINGET", "LONG_BINGET"]:
                        values.append(memo[int(ops[n - offset][1])])
                    elif ops[n - offset][0].name not in [
                        "SHORT_BINUNICODE",
                        "UNICODE",
                        "BINUNICODE",
                        "BINUNICODE8",
                    ]:
                        logger.debug(
                            "Presence of non-string opcode, categorizing as an unknown dangerous import"
                        )
                        values.append("unknown")
                    else:
                        values.append(ops[n - offset][1])
                    if len(values) == 2:
                        break
                if len(values) != 2:
                    raise ValueError(
                        f"Found {len(values)} values for STACK_GLOBAL at position {n} instead of 2."
                    )
                globals.add((values[1], values[0]))
        if parse_error is not None:
            raise GenOpsError(parse_error, globals or None)
        if not multiple_pickles:
            break
        try:
            last_byte = data.read(1)
            if last_byte:
                data.seek(-1, 1)
        except ValueError as exc:
            raise GenOpsError(str(exc), globals or None) from exc

    return globals


def scan_pickle_bytes(
    model: Model,
    settings: Dict[str, Any],
    scan_name: str = "pickle",
    multiple_pickles: bool = True,
    offset: int = 0,
) -> ScanResults:
    """Disassemble a Pickle stream and report issues"""
    issues: List[Issue] = []
    try:
        raw_globals = _list_globals(
            model.get_stream(offset), multiple_pickles, settings
        )
    except GenOpsError as e:
        if e.globals is not None:
            partial = _build_scan_result_from_raw_globals(e.globals, model, settings)
            partial.errors.append(
                PickleGenopsError(scan_name, f"Parsing error: {e}", model)
            )
            return partial
        return ScanResults(
            issues,
            [
                PickleGenopsError(
                    scan_name,
                    f"Parsing error: {e}",
                    model,
                )
            ],
            [],
        )
    logger.debug("Global imports in %s: %s", model, raw_globals, settings)
    return _build_scan_result_from_raw_globals(raw_globals, model, settings)


def _build_scan_result_from_raw_globals(
    raw_globals: Set[Tuple[str, str]],
    model: Model,
    settings: Dict[str, Any],
) -> ScanResults:
    issues: List[Issue] = []
    severities = {
        "CRITICAL": IssueSeverity.CRITICAL,
        "HIGH": IssueSeverity.HIGH,
        "MEDIUM": IssueSeverity.MEDIUM,
        "LOW": IssueSeverity.LOW,
    }

    for rg in raw_globals:
        global_module, global_name, severity = rg[0], rg[1], None
        for severity_name in severities:
            if global_module not in settings["unsafe_globals"][severity_name]:
                continue
            filter = settings["unsafe_globals"][severity_name][global_module]
            if filter == "*":
                severity = severities[severity_name]
                break
            for filter_value in filter:
                if filter_value in global_name:
                    severity = severities[severity_name]
                    break
            else:
                continue
            break
        if "unknown" in global_module or "unknown" in global_name:
            severity = IssueSeverity.CRITICAL  # we must assume it is RCE
        if severity is not None:
            issues.append(
                Issue(
                    code=IssueCode.UNSAFE_OPERATOR,
                    severity=severity,
                    details=OperatorIssueDetails(
                        module=global_module,
                        operator=global_name,
                        source=model.get_source(),
                        severity=severity,
                    ),
                )
            )
    return ScanResults(issues, [], [])


_MAX_NUMPY_HEADER_BYTES = 10_000


def _read_numpy_bytes(
    stream: IO[bytes], size: int, *, allow_eof: bool = False
) -> bytes:
    """Read a previously bounded field, including streams that return short reads."""
    if type(size) is not int or not 0 <= size <= _MAX_NUMPY_HEADER_BYTES:
        raise ValueError("NumPy read size is outside the header bound")
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not isinstance(chunk, bytes) or len(chunk) > remaining:
            raise ValueError("NumPy stream returned an invalid byte chunk")
        if not chunk:
            if allow_eof:
                break
            raise ValueError("Truncated NumPy header")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_numpy_array_header(
    stream: IO[bytes],
    version: Tuple[int, int],
    settings: Optional[Dict[str, Any]] = None,
) -> Any:
    """Validate an NPY header without letting its length control an unbounded read.

    Public NumPy readers cover versions 1 and 2, including their legacy literal
    compatibility. Version 3 requires UTF-8; reading it as version 2 corrupts
    Unicode field names. Only bounded literals and public dtype decoding are used.
    """
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError(f"Unsupported numpy file version: {version}")
    limits = _pickle_limits({} if settings is None else settings)
    maximum = min(
        _MAX_NUMPY_HEADER_BYTES,
        limits["max_pickle_argument_bytes"],
        limits["max_pickle_bytes"],
    )
    length_format = "<H" if version == (1, 0) else "<I"
    length_bytes = _read_numpy_bytes(stream, struct.calcsize(length_format))
    header_length = struct.unpack(length_format, length_bytes)[0]
    if not 0 < header_length <= maximum:
        raise ValueError(
            f"NumPy header byte limit exceeded or empty (maximum {maximum})"
        )
    header = _read_numpy_bytes(stream, header_length)
    try:
        if version in ((1, 0), (2, 0)):
            bounded = io.BytesIO(length_bytes + header)
            reader = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            shape, fortran_order, dtype = reader(bounded, max_header_size=maximum)
        else:
            expression = ast.parse(header.decode("utf-8").lstrip(" \t"), mode="eval")
            if not isinstance(expression.body, ast.Dict):
                raise ValueError("NumPy header must be a literal dictionary")
            names = []
            for key in expression.body.keys:
                if not isinstance(key, ast.Constant) or type(key.value) is not str:
                    raise ValueError("NumPy header keys must be literal strings")
                names.append(key.value)
            if len(names) != 3 or set(names) != {"descr", "fortran_order", "shape"}:
                raise ValueError(
                    "NumPy header keys are duplicate, missing, extra or invalid"
                )
            values = ast.literal_eval(expression)
            shape, fortran_order = values["shape"], values["fortran_order"]
            if not isinstance(values["descr"], (str, list, tuple)):
                raise ValueError("NumPy dtype descriptor is invalid")
            # The public API accepts string/tuple descriptors too; NumPy's
            # older type stubs list only structured-list descriptors.
            dtype = np.lib.format.descr_to_dtype(cast(Any, values["descr"]))
        if type(shape) is not tuple or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise ValueError("NumPy shape must contain non-negative integer dimensions")
        if type(fortran_order) is not bool:
            raise ValueError("NumPy fortran_order must be a boolean")
    except (
        SyntaxError,
        UnicodeError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as error:
        raise ValueError(f"Invalid NumPy header: {error}") from error
    return shape, fortran_order, dtype


def scan_numpy(model: Model, settings: Dict[str, Any]) -> ScanResults:
    """Inspect headers and object pickles; numeric tensor bodies are not loaded."""
    scan_name = "numpy"
    # Code to distinguish from NumPy binary files and pickles.
    _ZIP_PREFIX = b"PK\x03\x04"
    _ZIP_SUFFIX = b"PK\x05\x06"  # empty zip files start with this
    N = len(np.lib.format.MAGIC_PREFIX)
    stream = model.get_stream()
    magic = _read_numpy_bytes(stream, N, allow_eof=True)
    # If the file size is less than N, we need to make sure not
    # to seek past the beginning of the file
    stream.seek(-min(N, len(magic)), 1)  # back-up
    if magic.startswith(_ZIP_PREFIX) or magic.startswith(_ZIP_SUFFIX):
        # .npz file
        return ScanResults(
            [],
            [],
            [
                ModelScanSkipped(
                    scan_name,
                    SkipCategories.NOT_IMPLEMENTED,
                    "Scanning of .npz files is not implemented yet",
                    str(model.get_source()),
                )
            ],
        )

    elif magic == np.lib.format.MAGIC_PREFIX:
        # .npy file
        version = np.lib.format.read_magic(stream)
        _, _, dtype = _read_numpy_array_header(stream, version, settings)

        if dtype.hasobject:
            return scan_pickle_bytes(model, settings, scan_name, True, stream.tell())
        else:
            return ScanResults([], [], [])
    else:
        return scan_pickle_bytes(model, settings, scan_name)


def scan_pytorch(model: Model, settings: Dict[str, Any]) -> ScanResults:
    scan_name = "pytorch"
    should_read_directly = _should_read_directly(model.get_stream())
    if should_read_directly and model.get_stream().tell() == 0:
        # try loading from tar
        try:
            # TODO: implement loading from tar
            raise TarError()
        except TarError:
            # file does not contain a tar
            model.get_stream().seek(0)

    magic = get_magic_number(model.get_stream())
    if magic != MAGIC_NUMBER:
        return ScanResults(
            [],
            [],
            [
                ModelScanSkipped(
                    scan_name,
                    SkipCategories.MAGIC_NUMBER,
                    "Invalid magic number",
                    str(model.get_source()),
                )
            ],
        )

    return scan_pickle_bytes(model, settings, scan_name, multiple_pickles=False)

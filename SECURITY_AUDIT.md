# Security and production-readiness audit

Audit date: 2026-08-02
Scope: ModelScan source, archive and filesystem handling, tests, dependency
lock, packaging, and GitHub Actions in this downstream repository.

This repository is derived from the Protect AI ModelScan project. The changes
below are intentionally narrow and preserve the public API, scanner interfaces,
report format, and upstream file-format behavior where possible. This review is
not a certification that ModelScan, its third-party parsers, or scanned models
are safe.

## Executive result

The audit found high-impact resource-exhaustion and filesystem trust-boundary
gaps around untrusted directory trees and ZIP/Keras archives. It also found
shared mutable settings and CI/release supply-chain weaknesses. The
high-confidence fixes are implemented with regression coverage.

ModelScan remains a static detector. A clean scan cannot rule out model
backdoors, parser vulnerabilities, unsafe behavior, or malicious constructs
outside the signatures understood by the enabled scanners.

## Remediated findings

| Severity | Finding | Resolution |
|---|---|---|
| High | Root paths, directory entries, and model opens could follow links, accept special files, or change between enumeration and scan completion while retaining an apparent clean result. | Root and descendant paths use `lstat`/no-follow checks. Each single-link regular file is bound to its enumerated identity, checked against the opened descriptor and pathname after scanners run, and invalidated on change. Directory scans require matching stable pre/post identity manifests; a mismatch discards all apparent success for that tree. |
| High | Recursive directory discovery had no entry/file count, depth, path-length, per-file, or total-size limits. | Added deterministic, link-free traversal with configurable positive limits for each resource dimension. |
| High | ZIP handling bounded only selected members, accepted compression methods implicitly, and could let corrupt/partially-read members avoid CRC or decompression validation. | All central-directory entries are counted; duplicate, encrypted, overlong, oversized, over-total, compression-bomb, and unsupported-compression entries are rejected. Every file member is boundedly materialized before scanning, forcing decompression, declared-length, and CRC validation; all read/decompression failures use the existing fail-closed `BAD_ZIP` outcome. |
| Medium | Keras `config.json` could be decompressed and parsed without a dedicated bound. | Added a configurable size cap and strict UTF-8/JSON-object validation with parse failures reported as skipped unsafe input. |
| Medium | Shallow settings copies allowed one scanner instance or caller mutation to change nested defaults used by another instance. | ModelScan now deep-copies default and caller-provided settings for each instance. |
| Medium | Test dependencies contained versions with published vulnerabilities. | Lock updated to patched releases, including Requests, aiohttp, urllib3, setuptools, pip, torch, msgpack, and idna; `pip-audit` is part of CI. |
| Medium | The PyPI workflow used a long-lived API token and could publish a tag without running the repository's verification gates on that exact revision or proving that it came from `main`. | Publishing uses GitHub OIDC with a protected `pypi` environment. An annotated strict-SemVer tag without build metadata (for an unambiguous PyPI version) whose target is contained in `main` gates tests on Python 3.10–3.14 plus Black, strict mypy, Bandit, pip-audit, Gitleaks, lock validation, and package build. Only the uploaded artifact from that successful verification job can reach the dependent publish job. |
| Low | GitHub Actions used floating tag references and the test cache step referenced a missing step ID. | Action revisions are immutable full commit SHAs, job timeouts and least permissions are explicit, and the cache key now consumes a valid setup step output. |
| Low | Local tool versions had drifted from the lock and pre-commit used mutable tags. | Black and Poetry were aligned with current tooling; all pre-commit revisions are frozen to reviewed commit SHAs. |
| Low | CI lacked a repository secret-scanning gate. | Added an immutable-pinned Gitleaks history/worktree workflow. |

## Compatibility notes

- Existing CLI arguments, exit codes, result dictionaries, scanner settings,
  and report APIs are unchanged.
- Existing archive defaults remain permissive to reduce behavior changes. An
  operator can lower the limits without changing code.
- A symbolic link, hard-linked regular file, or special file that may previously
  have been followed is now returned as a path error. A ZIP rejected for a
  safety limit is reported through the existing skipped/bad-ZIP mechanism.
- ZIP compression is explicitly limited to runtime-supported Stored, Deflate,
  BZIP2, and LZMA methods. Corrupt data, unsupported methods, CRC failures, and
  decompression failures are reported as `BAD_ZIP` instead of clean scans.
- A file/tree identity change during a scan produces a path error and removes
  issues, scanned markers, and skip outcomes derived from that unstable input.
- Scanner modules and settings files are trusted configuration: specifying a
  Python import path intentionally imports and runs that module.

## Validation performed

- Python 3.12 project-local environment.
- 31 tests passing, including link, file/tree race, traversal-limit, ZIP duplicate,
  unsupported-compression, CRC-corruption, compression-ratio, invalid-limit,
  Keras-bound, and settings-isolation cases.
- Black check passing for 30 files.
- Mypy strict check passing for all 30 Python source and test files.
- Bandit passing with no reportable source findings.
- `pip-audit` reporting no known vulnerabilities in the installed
  Poetry-locked environment.
- Poetry configuration/lock consistency and sdist/wheel builds checked.
- GitHub Action references verified as full 40-character commit pins.
- Gitleaks history and worktree scans reporting no findings.
- Tag releases now require annotated strict SemVer without build metadata on
  `main`, test Python 3.10 through 3.14, and require the complete verification
  job before the exact uploaded artifacts can be published.

## Residual risks

| Severity | Residual risk | Production control |
|---|---|---|
| High | Optional h5py and TensorFlow scanners invoke large native/third-party parsers on attacker-controlled bytes. | Run scanning in a disposable, no-network, non-root worker with a read-only root/input, seccomp or equivalent syscall policy, and CPU/memory/PID/file/time limits. |
| High | Static signatures cannot prove that a model is benign or free of backdoors. | Require trusted publishers, immutable digests, signatures/provenance, independent validation, human approval for high-impact models, and runtime monitoring. |
| Medium | Python's ZIP reader parses the central directory before ModelScan can enforce entry limits. | Put a hard artifact-size and worker-memory limit outside the process; consider a streaming pre-parser for hostile archives. |
| Medium | File identities and before/after tree manifests detect ordinary replacement and mutation, but they are not an immutable filesystem snapshot and cannot defend against a privileged writer or hostile filesystem able to spoof metadata. | Make intake objects read-only to untrusted users and the scanner, scan an immutable digest-addressed object, and have the registry verify the same digest before admission. |
| Medium | Extension-based format selection can cause unsupported or disguised files to be skipped. | Treat skipped/unsupported results as policy failures for production admission and corroborate with independent format identification. |
| Medium | Custom scanner/reporting modules are arbitrary trusted Python code. | Permit only reviewed, pinned modules in an immutable worker image; never accept module paths from an untrusted requester. |
| Medium | The maximum compression-ratio default is compatibility-oriented and too high for many services. | Lower archive/member/count/ratio limits based on accepted formats and enforce an outer request-size limit. |
| Low | OIDC publishing still depends on correct external PyPI trusted-publisher configuration. | Bind the PyPI project to the exact repository, workflow, environment, and owner; require protected-environment approval. |

## SecAI_OS integration guidance

ModelScan should stay an upstream-compatible detection engine. SecAI_OS should
own the stronger admission policy and execute this tool behind its scanner
broker/worker isolation boundary. In particular:

1. Pin this package and every optional parser by lockfile and image digest.
2. Pass only a read-only artifact file into a fresh worker with no network.
3. Treat scanner errors, skipped files, and unsupported formats as fail-closed
   policy outcomes rather than clean results.
4. Record the artifact digest, policy digest, scanner package/image digest,
   result, and resource-limit termination status in signed provenance.
5. Keep a compatibility test matrix against upstream ModelScan before rebasing
   or contributing these changes upstream.

## Prioritized roadmap

### P0 — before production

- Add process-level timeout and byte-budget enforcement for every scanner.
- Add an explicit strict/admission mode that fails on any skipped or unsupported
  input while retaining current CLI behavior by default.
- Fuzz ZIP/Keras member handling, Pickle opcode parsing, HDF5 inputs, protobuf
  inputs, and path edge cases with a maintained regression corpus.
- Build and sign a minimal scanner worker image with SBOM and CVE gates.

### P1 — production operations

- Emit a stable machine-readable verdict schema with scanner/version evidence,
  limit violations, skips, and artifact digests.
- Add MIME/magic-based format corroboration and policy-configurable extension
  mismatch handling.
- Add structured, privacy-safe metrics for scan latency, limit rejections,
  skipped formats, dependency availability, and worker termination reasons.
- Automate upstream compatibility testing and downstream patch rebasing.

### P2 — feature improvements

- Add signed DSSE/in-toto scan attestations suitable for a model registry.
- Support scanner capability discovery so admission policies can require exact
  format coverage before accepting a model.
- Add incremental directory manifests and digest-based scan-result caching with
  scanner/policy-version invalidation.
- Offer an optional streaming archive preflight to reject hostile central
  directories before high-level ZIP processing.

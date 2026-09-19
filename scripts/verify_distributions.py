"""Fail closed when MosaicFeed release archives contain unexpected files."""

from __future__ import annotations

import argparse
import stat
import tarfile
import tomllib
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

MAX_SDIST_FILES = 1_000
MAX_SDIST_BYTES = 10 * 1024 * 1024
MAX_WHEEL_FILES = 200
MAX_WHEEL_BYTES = 5 * 1024 * 1024

SDIST_TOP_LEVEL = frozenset(
    {
        ".github",
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "SECURITY.md",
        "docs",
        "examples",
        "pyproject.toml",
        "scripts",
        "src",
        "tests",
        "uv.lock",
    }
)
WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
    | {f"COM{index}" for index in "¹²³"}
    | {f"LPT{index}" for index in "¹²³"}
)


def _safe_parts(name: str) -> tuple[str, ...]:
    raw_parts = name.split("/")
    path = PurePosixPath(name)
    unsafe = (
        "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(
            ":" in part
            or part.endswith((" ", "."))
            or part.split(".", maxsplit=1)[0].upper() in WINDOWS_DEVICE_NAMES
            for part in raw_parts
        )
    )
    if unsafe:
        raise ValueError(f"archive contains an unsafe path: {name!r}")
    return path.parts


def _require_unique_portable_paths(names: list[str]) -> None:
    keys = [unicodedata.normalize("NFC", name).casefold() for name in names]
    if len(keys) != len(set(keys)):
        raise ValueError("archive contains paths that collide on a portable filesystem")


def _metadata_value(contents: bytes, field: str) -> str:
    prefix = f"{field}: ".encode()
    values = [
        line[len(prefix) :].decode("utf-8")
        for line in contents.splitlines()
        if line.startswith(prefix)
    ]
    if len(values) != 1:
        raise ValueError(f"archive metadata must contain exactly one {field} field")
    return values[0]


def verify_sdist(path: Path, *, version: str) -> None:
    expected_root = f"mosaicfeed-{version}"
    required = {
        f"{expected_root}/PKG-INFO",
        f"{expected_root}/docs/event-stream.md",
        f"{expected_root}/docs/http-inference.md",
        f"{expected_root}/scripts/verify_distributions.py",
        f"{expected_root}/scripts/verify_branch_coverage.py",
        f"{expected_root}/src/mosaicfeed/event_stream.py",
        f"{expected_root}/src/mosaicfeed/server.py",
    }
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("source distribution contains duplicate paths")
        _require_unique_portable_paths(names)
        if len(members) > MAX_SDIST_FILES:
            raise ValueError("source distribution exceeds its file-count ceiling")
        total_bytes = 0
        for member in members:
            parts = _safe_parts(member.name)
            if parts[0] != expected_root or len(parts) < 2:
                raise ValueError(f"unexpected source-distribution root: {member.name!r}")
            if parts[1] not in SDIST_TOP_LEVEL:
                raise ValueError(f"unexpected source-distribution member: {member.name!r}")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported source-distribution member: {member.name!r}")
            if member.isfile():
                total_bytes += member.size
        if total_bytes > MAX_SDIST_BYTES:
            raise ValueError("source distribution exceeds its uncompressed byte ceiling")
        missing = required - set(names)
        if missing:
            raise ValueError(f"source distribution is missing required files: {sorted(missing)}")
        metadata = archive.extractfile(f"{expected_root}/PKG-INFO")
        if metadata is None:
            raise ValueError("source distribution PKG-INFO is not a regular file")
        metadata_bytes = metadata.read()
    if _metadata_value(metadata_bytes, "Name") != "mosaicfeed":
        raise ValueError("source distribution has the wrong project name")
    if _metadata_value(metadata_bytes, "Version") != version:
        raise ValueError("source distribution has the wrong version")


def verify_wheel(path: Path, *, version: str) -> None:
    expected_dist_info = f"mosaicfeed-{version}.dist-info"
    required = {
        "mosaicfeed/event_stream.py",
        "mosaicfeed/py.typed",
        "mosaicfeed/server.py",
        f"{expected_dist_info}/METADATA",
        f"{expected_dist_info}/RECORD",
        f"{expected_dist_info}/WHEEL",
    }
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate paths")
        _require_unique_portable_paths(names)
        if len(infos) > MAX_WHEEL_FILES:
            raise ValueError("wheel exceeds its file-count ceiling")
        total_bytes = 0
        for info in infos:
            parts = _safe_parts(info.filename)
            if parts[0] not in {"mosaicfeed", expected_dist_info}:
                raise ValueError(f"unexpected wheel member: {info.filename!r}")
            file_type = stat.S_IFMT(info.external_attr >> 16)
            expected_types = {0, stat.S_IFDIR} if info.is_dir() else {0, stat.S_IFREG}
            if file_type not in expected_types:
                raise ValueError(f"wheel contains a special filesystem entry: {info.filename!r}")
            if "__pycache__" in parts or info.filename.endswith((".pyc", ".pyo")):
                raise ValueError(f"wheel contains generated bytecode: {info.filename!r}")
            total_bytes += info.file_size
        if total_bytes > MAX_WHEEL_BYTES:
            raise ValueError("wheel exceeds its uncompressed byte ceiling")
        missing = required - set(names)
        if missing:
            raise ValueError(f"wheel is missing required files: {sorted(missing)}")
        metadata_bytes = archive.read(f"{expected_dist_info}/METADATA")
    if _metadata_value(metadata_bytes, "Name") != "mosaicfeed":
        raise ValueError("wheel has the wrong project name")
    if _metadata_value(metadata_bytes, "Version") != version:
        raise ValueError("wheel has the wrong version")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    with Path("pyproject.toml").open("rb") as source:
        version = tomllib.load(source)["project"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("project.version must be a non-empty string")
    verify_sdist(args.dist / f"mosaicfeed-{version}.tar.gz", version=version)
    verify_wheel(args.dist / f"mosaicfeed-{version}-py3-none-any.whl", version=version)
    print(f"verified bounded allowlisted distributions for MosaicFeed {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

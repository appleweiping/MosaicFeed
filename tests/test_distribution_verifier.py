from __future__ import annotations

import io
import runpy
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

VERIFIER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "verify_distributions.py"))
_metadata_value = VERIFIER["_metadata_value"]
_require_unique_portable_paths = VERIFIER["_require_unique_portable_paths"]
_safe_parts = VERIFIER["_safe_parts"]
verify_sdist = VERIFIER["verify_sdist"]
verify_wheel = VERIFIER["verify_wheel"]


def _sdist_files(version: str) -> dict[str, bytes]:
    root = f"mosaicfeed-{version}"
    return {
        f"{root}/PKG-INFO": f"Name: mosaicfeed\nVersion: {version}\n".encode(),
        f"{root}/docs/event-stream.md": b"events\n",
        f"{root}/docs/http-inference.md": b"http\n",
        f"{root}/docs/listwise-impressions.md": b"listwise\n",
        f"{root}/docs/mind-submission.md": b"mind submission\n",
        f"{root}/docs/neural-news.md": b"neural\n",
        f"{root}/docs/neural-news-selection.md": b"selection\n",
        f"{root}/docs/text-features.md": b"text\n",
        f"{root}/examples/text-events.json": b"[]\n",
        f"{root}/examples/text-vocabulary-articles.json": b"[]\n",
        f"{root}/examples/neural_news_train.json": b"[]\n",
        f"{root}/examples/neural_news_validation.json": b"[]\n",
        f"{root}/examples/neural_news_selection_plan.json": b"{}\n",
        f"{root}/examples/mind_truth.txt": b"example [0,1]\n",
        f"{root}/scripts/verify_distributions.py": b"# verifier\n",
        f"{root}/scripts/verify_branch_coverage.py": b"# coverage gate\n",
        f"{root}/src/mosaicfeed/event_stream.py": b"# events\n",
        f"{root}/src/mosaicfeed/listwise.py": b"# listwise\n",
        f"{root}/src/mosaicfeed/neural_news.py": b"# neural\n",
        f"{root}/src/mosaicfeed/neural_news_selection.py": b"# selection\n",
        f"{root}/src/mosaicfeed/mind_submission.py": b"# submission\n",
        f"{root}/src/mosaicfeed/server.py": b"# server\n",
        f"{root}/src/mosaicfeed/text_features.py": b"# text\n",
        f"{root}/tests/test_text_features.py": b"# text tests\n",
        f"{root}/tests/test_listwise.py": b"# listwise tests\n",
        f"{root}/tests/test_neural_news.py": b"# neural tests\n",
        f"{root}/tests/test_neural_news_selection.py": b"# selection tests\n",
        f"{root}/tests/test_mind_submission.py": b"# submission tests\n",
    }


def _write_sdist(
    path: Path,
    files: dict[str, bytes],
    *,
    extra_members: tuple[tarfile.TarInfo, ...] = (),
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
        for info in extra_members:
            archive.addfile(info)


def _wheel_files(version: str) -> dict[str, bytes]:
    metadata = f"mosaicfeed-{version}.dist-info"
    return {
        "mosaicfeed/event_stream.py": b"# events\n",
        "mosaicfeed/listwise.py": b"# listwise\n",
        "mosaicfeed/neural_news.py": b"# neural\n",
        "mosaicfeed/neural_news_selection.py": b"# selection\n",
        "mosaicfeed/mind_submission.py": b"# submission\n",
        "mosaicfeed/py.typed": b"",
        "mosaicfeed/server.py": b"# server\n",
        "mosaicfeed/text_features.py": b"# text\n",
        f"{metadata}/METADATA": f"Name: mosaicfeed\nVersion: {version}\n".encode(),
        f"{metadata}/RECORD": b"",
        f"{metadata}/WHEEL": b"Wheel-Version: 1.0\n",
    }


def _write_wheel(
    path: Path,
    files: dict[str, bytes],
    *,
    special: tuple[str, int] | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
        if special is not None:
            name, file_type = special
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (file_type | 0o644) << 16
            archive.writestr(info, b"special")


@pytest.mark.parametrize(
    "name",
    (
        "../escape",
        "/absolute",
        "root//double",
        "root/./dot",
        "root/../escape",
        "root\\windows-separator",
        "root/nul\0suffix",
        "root/control\nsuffix",
        "root/trailing-dot.",
        "root/trailing-space ",
        "root/drive:C",
        "root/CON",
        "root/aux.txt",
        "root/CONIN$.txt",
        "root/conout$",
        "root/COM¹.txt",
        "root/lpt³",
    ),
)
def test_archive_path_validation_rejects_noncanonical_or_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe path"):
        _safe_parts(name)


def test_archive_metadata_requires_one_exact_field() -> None:
    assert _metadata_value(b"Name: mosaicfeed\nVersion: 0.5.0\n", "Name") == "mosaicfeed"
    with pytest.raises(ValueError, match="exactly one"):
        _metadata_value(b"Name: first\nName: second\n", "Name")
    with pytest.raises(ValueError, match="exactly one"):
        _metadata_value(b"Version: 0.5.0\n", "Name")


@pytest.mark.parametrize(
    "names",
    (
        ["root/module.py", "ROOT/MODULE.PY"],
        [
            "root/caf\N{LATIN SMALL LETTER E WITH ACUTE}.py",
            "root/cafe\N{COMBINING ACUTE ACCENT}.py",
        ],
    ),
)
def test_archive_paths_reject_cross_platform_collisions(names: list[str]) -> None:
    with pytest.raises(ValueError, match="portable filesystem"):
        _require_unique_portable_paths(names)


def test_minimal_valid_distribution_archives_pass(tmp_path: Path) -> None:
    version = "0.5.0"
    sdist = tmp_path / "mosaicfeed-0.5.0.tar.gz"
    wheel = tmp_path / "mosaicfeed-0.5.0-py3-none-any.whl"
    _write_sdist(sdist, _sdist_files(version))
    _write_wheel(wheel, _wheel_files(version))

    verify_sdist(sdist, version=version)
    verify_wheel(wheel, version=version)


def test_text_feature_slice_is_required_in_both_archives(tmp_path: Path) -> None:
    version = "0.5.0"
    sdist_files = _sdist_files(version)
    del sdist_files[f"mosaicfeed-{version}/docs/text-features.md"]
    sdist = tmp_path / "without-text-docs.tar.gz"
    _write_sdist(sdist, sdist_files)
    with pytest.raises(ValueError, match="missing required files"):
        verify_sdist(sdist, version=version)

    wheel_files = _wheel_files(version)
    del wheel_files["mosaicfeed/text_features.py"]
    wheel = tmp_path / "without-text-code.whl"
    _write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="missing required files"):
        verify_wheel(wheel, version=version)


def test_submission_slice_is_required_in_both_archives(tmp_path: Path) -> None:
    version = "0.5.0"
    sdist_files = _sdist_files(version)
    del sdist_files[f"mosaicfeed-{version}/docs/mind-submission.md"]
    sdist = tmp_path / "without-submission-docs.tar.gz"
    _write_sdist(sdist, sdist_files)
    with pytest.raises(ValueError, match="missing required files"):
        verify_sdist(sdist, version=version)

    wheel_files = _wheel_files(version)
    del wheel_files["mosaicfeed/mind_submission.py"]
    wheel = tmp_path / "without-submission-code.whl"
    _write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="missing required files"):
        verify_wheel(wheel, version=version)


def test_listwise_slice_is_required_in_both_archives(tmp_path: Path) -> None:
    version = "0.5.0"
    sdist_files = _sdist_files(version)
    del sdist_files[f"mosaicfeed-{version}/src/mosaicfeed/listwise.py"]
    sdist = tmp_path / "without-listwise-code.tar.gz"
    _write_sdist(sdist, sdist_files)
    with pytest.raises(ValueError, match="missing required files"):
        verify_sdist(sdist, version=version)

    wheel_files = _wheel_files(version)
    del wheel_files["mosaicfeed/listwise.py"]
    wheel = tmp_path / "without-listwise-code.whl"
    _write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="missing required files"):
        verify_wheel(wheel, version=version)


def test_neural_news_slice_is_required_in_both_archives(tmp_path: Path) -> None:
    version = "0.5.0"
    sdist_files = _sdist_files(version)
    del sdist_files[f"mosaicfeed-{version}/docs/neural-news.md"]
    sdist = tmp_path / "without-neural-docs.tar.gz"
    _write_sdist(sdist, sdist_files)
    with pytest.raises(ValueError, match="missing required files"):
        verify_sdist(sdist, version=version)

    wheel_files = _wheel_files(version)
    del wheel_files["mosaicfeed/neural_news.py"]
    wheel = tmp_path / "without-neural-code.whl"
    _write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="missing required files"):
        verify_wheel(wheel, version=version)


def test_real_archives_reject_extended_windows_device_names(tmp_path: Path) -> None:
    version = "0.5.0"
    wheel = tmp_path / "malicious.whl"
    wheel_files = _wheel_files(version)
    wheel_files["mosaicfeed/COM¹.txt"] = b"unsafe"
    _write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="unsafe path"):
        verify_wheel(wheel, version=version)

    sdist = tmp_path / "malicious.tar.gz"
    member = tarfile.TarInfo(f"mosaicfeed-{version}/src/CONOUT$.txt")
    _write_sdist(sdist, _sdist_files(version), extra_members=(member,))
    with pytest.raises(ValueError, match="unsafe path"):
        verify_sdist(sdist, version=version)


def test_sdist_rejects_duplicate_and_link_members(tmp_path: Path) -> None:
    version = "0.5.0"
    root = f"mosaicfeed-{version}"
    duplicate_path = tmp_path / "duplicate.tar.gz"
    duplicate = tarfile.TarInfo(f"{root}/docs/event-stream.md")
    _write_sdist(duplicate_path, _sdist_files(version), extra_members=(duplicate,))
    with pytest.raises(ValueError, match="duplicate"):
        verify_sdist(duplicate_path, version=version)

    link_path = tmp_path / "link.tar.gz"
    link = tarfile.TarInfo(f"{root}/src/mosaicfeed/alias.py")
    link.type = tarfile.SYMTYPE
    link.linkname = "server.py"
    _write_sdist(link_path, _sdist_files(version), extra_members=(link,))
    with pytest.raises(ValueError, match="unsupported"):
        verify_sdist(link_path, version=version)


def test_wheel_rejects_special_and_portably_colliding_members(tmp_path: Path) -> None:
    version = "0.5.0"
    special_path = tmp_path / "special.whl"
    _write_wheel(
        special_path,
        _wheel_files(version),
        special=("mosaicfeed/pipe", stat.S_IFIFO),
    )
    with pytest.raises(ValueError, match="special filesystem entry"):
        verify_wheel(special_path, version=version)

    collision_path = tmp_path / "collision.whl"
    files = _wheel_files(version)
    files["MOSAICFEED/SERVER.PY"] = b"collision"
    _write_wheel(collision_path, files)
    with pytest.raises(ValueError, match="portable filesystem"):
        verify_wheel(collision_path, version=version)

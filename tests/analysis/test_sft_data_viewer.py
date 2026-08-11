from io import BytesIO
from pathlib import Path

import pytest

from analysis.sft_data_viewer import sft_data_viewer


@pytest.fixture
def viewer(tmp_path):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    sft_data_viewer.app.config.update(
        TESTING=True,
        SAFE_ROOT=safe_root,
        UPLOAD_FOLDER=str(upload_root),
    )
    sft_data_viewer.clear_current_file()
    sft_data_viewer.total_records = 0
    yield sft_data_viewer.app.test_client(), safe_root
    sft_data_viewer.clear_current_file()


def _write_jsonl(path: Path):
    path.write_text('{"messages": [], "problem": "test"}\n', encoding="utf-8")


def test_load_file_accepts_in_root_jsonl(viewer):
    client, safe_root = viewer
    path = safe_root / "records.jsonl"
    _write_jsonl(path)

    response = client.post("/load_file", data={"filepath": str(path)})

    assert response.status_code == 302


def test_upload_does_not_follow_preexisting_filename_symlink(viewer, tmp_path):
    client, _ = viewer
    outside = tmp_path / "outside.jsonl"
    outside.write_text("unchanged", encoding="utf-8")
    link = Path(sft_data_viewer.app.config["UPLOAD_FOLDER"]) / "records.jsonl"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    response = client.post(
        "/upload",
        data={
            "file": (
                BytesIO(b'{"problem": "inside", "messages": []}\n'),
                "records.jsonl",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 302
    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_load_file_rejects_traversal_and_absolute_outside_paths(viewer, tmp_path):
    client, safe_root = viewer
    outside = tmp_path / "outside.jsonl"
    _write_jsonl(outside)

    for path in ("../outside.jsonl", str(outside)):
        response = client.post("/load_file", data={"filepath": path})

        assert response.status_code in {400, 403, 404}
        assert str(outside) not in response.get_data(as_text=True)


def test_load_file_rejects_sibling_prefix_path(viewer):
    client, safe_root = viewer
    sibling = safe_root.parent / f"{safe_root.name}-sibling"
    sibling.mkdir()
    path = sibling / "records.jsonl"
    _write_jsonl(path)

    response = client.post("/load_file", data={"filepath": str(path)})

    assert response.status_code in {400, 403, 404}


def test_load_file_rejects_wrong_suffix(viewer):
    client, safe_root = viewer
    path = safe_root / "records.json"
    path.write_text("{}", encoding="utf-8")

    response = client.post("/load_file", data={"filepath": str(path)})

    assert response.status_code == 400


def test_load_file_rejects_symlink_escape(viewer, tmp_path):
    client, safe_root = viewer
    outside = tmp_path / "outside.jsonl"
    _write_jsonl(outside)
    link = safe_root / "linked.jsonl"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    response = client.post("/load_file", data={"filepath": str(link)})

    assert response.status_code in {400, 403, 404}


def test_loaded_file_cannot_be_swapped_for_outside_symlink(viewer, tmp_path):
    client, safe_root = viewer
    path = safe_root / "records.jsonl"
    path.write_text('{"problem": "inside", "messages": []}\n', encoding="utf-8")
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"problem": "outside", "messages": []}\n', encoding="utf-8")
    assert client.post("/load_file", data={"filepath": str(path)}).status_code == 302

    try:
        path.unlink()
        path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Open-file replacement is unavailable: {exc}")

    response = client.get("/api/record/0")

    assert response.status_code == 200
    assert response.get_json()["problem"] == "inside"


def test_load_file_rejects_hard_link_to_outside_file(viewer, tmp_path):
    client, safe_root = viewer
    outside = tmp_path / "outside.jsonl"
    _write_jsonl(outside)
    hard_link = safe_root / "hard-linked.jsonl"
    try:
        hard_link.hardlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Hard links are unavailable: {exc}")

    response = client.post("/load_file", data={"filepath": str(hard_link)})

    assert response.status_code == 403


def test_untrusted_host_is_rejected(viewer):
    client, safe_root = viewer
    path = safe_root / "records.jsonl"
    _write_jsonl(path)

    response = client.post(
        "/load_file",
        data={"filepath": str(path)},
        headers={"Host": "evil.example"},
    )

    assert response.status_code == 400


def test_server_defaults_to_loopback_and_current_directory():
    assert sft_data_viewer.DEFAULT_HOST == "127.0.0.1"
    assert sft_data_viewer.DEFAULT_SAFE_ROOT == Path.cwd().resolve()

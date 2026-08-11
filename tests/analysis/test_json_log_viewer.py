import json
from io import BytesIO
from pathlib import Path

import pytest

from analysis.json_log_viewer import json_log_viewer


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(json_log_viewer, "SAFE_ROOT", str(safe_root))
    json_log_viewer.app.config.update(
        TESTING=True,
        SAFE_ROOT=safe_root,
        UPLOAD_FOLDER=str(upload_root),
        ALLOWED_ORIGINS=(),
    )
    json_log_viewer.data = None
    json_log_viewer.current_file = None
    return json_log_viewer.app.test_client(), safe_root


def _write_log(path: Path):
    path.write_text(
        json.dumps(
            {
                "problem": "test",
                "config": {},
                "uuid": "test",
                "success": True,
                "log": [{"action": None}],
            }
        ),
        encoding="utf-8",
    )


def test_load_file_from_path_accepts_in_root_json(viewer):
    client, safe_root = viewer
    path = safe_root / "trajectory.json"
    _write_log(path)

    response = client.get("/load_file_from_path", query_string={"path": str(path)})

    assert response.status_code == 200
    assert response.get_json()["success"] is True


def test_upload_does_not_follow_preexisting_filename_symlink(viewer, tmp_path):
    client, _ = viewer
    outside = tmp_path / "outside.json"
    outside.write_text("unchanged", encoding="utf-8")
    link = Path(json_log_viewer.app.config["UPLOAD_FOLDER"]) / "trajectory.json"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")
    payload = json.dumps(
        {
            "problem": "test",
            "config": {},
            "uuid": "test",
            "success": True,
            "log": [],
        }
    ).encode()

    response = client.post(
        "/upload",
        data={"file": (BytesIO(payload), "trajectory.json")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 302
    assert outside.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize("route", ["/load_file_from_path", "/browse_directory"])
def test_routes_reject_traversal_and_absolute_outside_paths(viewer, tmp_path, route):
    client, safe_root = viewer
    outside = tmp_path / "outside.json"
    _write_log(outside)

    for path in ("../outside.json", str(outside)):
        response = client.get(route, query_string={"path": path})

        assert response.status_code in {400, 403, 404}
        assert str(outside) not in response.get_data(as_text=True)


@pytest.mark.parametrize("route", ["/load_file_from_path", "/browse_directory"])
def test_routes_reject_sibling_prefix_path(viewer, tmp_path, route):
    client, safe_root = viewer
    sibling = safe_root.parent / f"{safe_root.name}-sibling"
    sibling.mkdir()
    path = sibling / "trajectory.json"
    _write_log(path)

    response = client.get(route, query_string={"path": str(path)})

    assert response.status_code in {400, 403, 404}


def test_load_file_from_path_rejects_wrong_suffix(viewer):
    client, safe_root = viewer
    path = safe_root / "trajectory.txt"
    path.write_text("{}", encoding="utf-8")

    response = client.get("/load_file_from_path", query_string={"path": str(path)})

    assert response.status_code == 400


@pytest.mark.parametrize("route", ["/load_file_from_path", "/browse_directory"])
def test_routes_reject_symlink_escape(viewer, tmp_path, route):
    client, safe_root = viewer
    outside = tmp_path / "outside.json"
    _write_log(outside)
    link = safe_root / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")

    response = client.get(route, query_string={"path": str(link)})

    assert response.status_code in {400, 403, 404}


def test_browse_directory_accepts_safe_root(viewer):
    client, safe_root = viewer
    _write_log(safe_root / "trajectory.json")

    response = client.get("/browse_directory", query_string={"path": str(safe_root)})

    assert response.status_code == 200
    assert response.get_json()["current_path"] == str(safe_root.resolve())


def test_cors_is_limited_to_explicit_allowed_origins(viewer):
    client, safe_root = viewer
    path = safe_root / "trajectory.json"
    _write_log(path)
    json_log_viewer.app.config["ALLOWED_ORIGINS"] = ("https://gray.example",)

    denied = client.get(
        "/load_file_from_path",
        query_string={"path": str(path)},
        headers={"Origin": "https://evil.example"},
    )
    allowed = client.get(
        "/load_file_from_path",
        query_string={"path": str(path)},
        headers={"Origin": "https://gray.example"},
    )

    assert "Access-Control-Allow-Origin" not in denied.headers
    assert allowed.headers["Access-Control-Allow-Origin"] == "https://gray.example"


def test_untrusted_host_cannot_bypass_origin_allowlist(viewer):
    client, safe_root = viewer
    path = safe_root / "trajectory.json"
    _write_log(path)

    response = client.get(
        "/load_file_from_path",
        query_string={"path": str(path)},
        headers={
            "Host": "evil.example",
            "Origin": "http://evil.example",
        },
    )

    assert response.status_code == 400


def test_server_defaults_to_loopback_and_current_directory():
    assert json_log_viewer.DEFAULT_HOST == "127.0.0.1"
    assert json_log_viewer.DEFAULT_SAFE_ROOT == Path.cwd().resolve()

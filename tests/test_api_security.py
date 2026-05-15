"""
tests/test_api_security.py
Tests for the new security layer in api/security.py + api/server.py:
  - API-key auth (dependency)
  - CORS allowlist parsing
  - safe_path_within + sanitise_filename helpers
  - End-to-end via FastAPI TestClient: auth-protected routes reject
    missing/invalid keys; path-traversal payloads are refused; the new
    PATCH /hypothesis/{id}/notes endpoint round-trips.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from api.security import (
    get_cors_origins,
    safe_path_within,
    sanitise_filename,
)

# ===========================================================================
# Pure helpers
# ===========================================================================

class TestSanitiseFilename:
    def test_basic_csv(self):
        assert sanitise_filename("data.csv", (".csv",)) == "data.csv"

    def test_strips_path_components(self):
        assert sanitise_filename("../etc/passwd.csv", (".csv",)) == "passwd.csv"

    def test_strips_backslash_path(self):
        assert sanitise_filename("C:\\\\Windows\\\\evil.csv", (".csv",)) == "evil.csv"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            sanitise_filename("", (".csv",))

    def test_rejects_dot_only(self):
        with pytest.raises(ValueError):
            sanitise_filename(".", (".csv",))

    def test_rejects_dotdot(self):
        with pytest.raises(ValueError):
            sanitise_filename("..", (".csv",))

    def test_rejects_wrong_extension(self):
        with pytest.raises(ValueError, match="not allowed"):
            sanitise_filename("script.py", (".csv",))

    def test_extension_is_case_insensitive(self):
        assert sanitise_filename("DATA.CSV", (".csv",)) == "DATA.CSV"

    def test_allows_multiple_extensions(self):
        assert sanitise_filename("foo.tsv", (".csv", ".tsv")) == "foo.tsv"

    def test_rejects_control_chars(self):
        with pytest.raises(ValueError):
            sanitise_filename("ev\x00il.csv", (".csv",))


class TestSafePathWithin:
    def test_inside_root_ok(self, tmp_path: Path):
        target = tmp_path / "ok.csv"
        target.write_text("x")
        resolved = safe_path_within(str(target), str(tmp_path))
        assert resolved == target.resolve()

    def test_escape_via_dotdot(self, tmp_path: Path):
        evil = tmp_path / "sub" / ".." / ".." / "outside.csv"
        with pytest.raises(ValueError, match="outside"):
            safe_path_within(str(evil), str(tmp_path / "sub"))

    def test_absolute_path_outside(self, tmp_path: Path):
        outside = tmp_path.parent / "outside.csv"
        with pytest.raises(ValueError, match="outside"):
            safe_path_within(str(outside), str(tmp_path))

    def test_root_is_created_if_missing(self, tmp_path: Path):
        new_root = tmp_path / "fresh"
        assert not new_root.exists()
        # Resolving a file inside the new root should succeed and create it.
        safe_path_within(str(new_root / "x.csv"), str(new_root))
        assert new_root.is_dir()


class TestGetCorsOrigins:
    def test_default_dev_list_when_unset(self, monkeypatch):
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        origins = get_cors_origins()
        assert "http://localhost:3000" in origins
        assert "http://localhost:5173" in origins
        assert "*" not in origins

    def test_explicit_allowlist(self, monkeypatch):
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.com,https://staging.example.com")
        origins = get_cors_origins()
        assert origins == ["https://app.example.com", "https://staging.example.com"]

    def test_wildcard_refused(self, monkeypatch):
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
        origins = get_cors_origins()
        # Falls back to dev list, NEVER returns ["*"] with credentials.
        assert "*" not in origins
        assert any("localhost" in o for o in origins)


# ===========================================================================
# FastAPI integration via TestClient
# ===========================================================================

@pytest.fixture
def authed_client(monkeypatch, tmp_path):
    """A TestClient with API_KEYS set so we can exercise both paths."""
    pytest.importorskip("fastapi")
    monkeypatch.setenv("API_KEYS", "test-key-123")
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED", raising=False)
    # Isolate uploads/sessions in tmp_path so tests can't write into the repo.
    # We reach in and rebind the module globals — the constants are computed
    # at import time but server.py reads them as module attributes each call.
    from fastapi.testclient import TestClient

    from api import server as srv
    monkeypatch.setattr(srv, "UPLOAD_DIR", str(tmp_path / "uploads"), raising=True)
    monkeypatch.setattr(srv, "SESSIONS_DIR", str(tmp_path / "sessions"), raising=True)
    os.makedirs(srv.UPLOAD_DIR, exist_ok=True)
    os.makedirs(srv.SESSIONS_DIR, exist_ok=True)
    return TestClient(srv.app)


class TestAuthIntegration:
    def test_health_endpoint_is_public(self, authed_client):
        r = authed_client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "online"

    def test_protected_endpoint_rejects_missing_key(self, authed_client):
        r = authed_client.get("/session/state")
        assert r.status_code == 401

    def test_protected_endpoint_rejects_bad_key(self, authed_client):
        r = authed_client.get("/session/state", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_protected_endpoint_accepts_good_key(self, authed_client):
        r = authed_client.get("/session/state", headers={"X-API-Key": "test-key-123"})
        assert r.status_code == 200


class TestUploadPathTraversal:
    def test_basename_dotdot_is_neutralised(self, authed_client, tmp_path):
        # ../../../etc/passwd → just "passwd" after sanitisation, which
        # then fails the extension check.
        files = {"file": ("../../../etc/passwd", b"col1,col2\n1,2", "text/plain")}
        r = authed_client.post(
            "/upload/csv",
            files=files,
            headers={"X-API-Key": "test-key-123"},
        )
        # The sanitiser strips path components first, then refuses extension.
        assert r.status_code == 400

    def test_wrong_extension_rejected(self, authed_client):
        files = {"file": ("evil.py", b"print('owned')", "text/plain")}
        r = authed_client.post(
            "/upload/csv",
            files=files,
            headers={"X-API-Key": "test-key-123"},
        )
        assert r.status_code == 400
        assert "not allowed" in r.json()["detail"]

    def test_csv_upload_succeeds(self, authed_client, tmp_path):
        files = {"file": ("data.csv", b"col1,col2\n1,2", "text/csv")}
        r = authed_client.post(
            "/upload/csv",
            files=files,
            headers={"X-API-Key": "test-key-123"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["filename"] == "data.csv"
        # The file must live inside the test UPLOAD_DIR — never outside.
        from api import server as srv
        resolved = Path(body["path"]).resolve()
        assert str(resolved).startswith(str(Path(srv.UPLOAD_DIR).resolve()))


class TestUpdateHypothesisNotes:
    """The endpoint previously crashed because the orchestrator method
    didn't exist; verify the round-trip works."""

    def test_404_on_unknown_id(self, authed_client):
        r = authed_client.patch(
            "/hypothesis/does-not-exist/notes",
            params={"notes": "hello"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert r.status_code == 404

    def test_round_trip_sets_notes(self, authed_client):
        from api import server as srv
        from models.hypothesis import Hypothesis
        hyp = Hypothesis(id="h1", title="Test")
        srv.scientist.context_memory.hypotheses[hyp.id] = hyp
        try:
            r = authed_client.patch(
                "/hypothesis/h1/notes",
                params={"notes": "looks suspicious — recheck Fig.3"},
                headers={"X-API-Key": "test-key-123"},
            )
            assert r.status_code == 200
            assert r.json()["scientist_notes"] == "looks suspicious — recheck Fig.3"
            assert hyp.scientist_notes == "looks suspicious — recheck Fig.3"
        finally:
            srv.scientist.context_memory.hypotheses.pop("h1", None)


class TestAnalysisFilePathSafety:
    def test_traversal_path_rejected(self, authed_client):
        from api import server as srv
        from models.hypothesis import Hypothesis
        hyp = Hypothesis(id="h2", title="Test")
        srv.scientist.context_memory.hypotheses[hyp.id] = hyp
        try:
            # An absolute path outside UPLOAD_DIR must be rejected.
            r = authed_client.post(
                "/workflow/analysis/h2",
                json={"file_path": "/etc/passwd"},
                headers={"X-API-Key": "test-key-123"},
            )
            assert r.status_code == 400
            assert "outside" in r.json()["detail"].lower()
        finally:
            srv.scientist.context_memory.hypotheses.pop("h2", None)


# ===========================================================================
# SSRF protection in rag_system
# ===========================================================================

class TestPdfUrlSafety:
    def test_arxiv_https_ok(self):
        from rag_system import _is_pdf_url_safe
        assert _is_pdf_url_safe("https://arxiv.org/pdf/2103.12345.pdf")
        assert _is_pdf_url_safe("https://www.arxiv.org/pdf/2103.12345.pdf")

    def test_http_refused(self):
        from rag_system import _is_pdf_url_safe
        assert not _is_pdf_url_safe("http://arxiv.org/pdf/2103.12345.pdf")

    def test_file_scheme_refused(self):
        from rag_system import _is_pdf_url_safe
        assert not _is_pdf_url_safe("file:///etc/passwd")

    def test_aws_metadata_refused(self):
        from rag_system import _is_pdf_url_safe
        assert not _is_pdf_url_safe("http://169.254.169.254/latest/meta-data/")
        assert not _is_pdf_url_safe("https://169.254.169.254/latest/meta-data/")

    def test_arbitrary_host_refused(self):
        from rag_system import _is_pdf_url_safe
        assert not _is_pdf_url_safe("https://attacker.example.com/x.pdf")

    def test_userinfo_refused(self):
        from rag_system import _is_pdf_url_safe
        # https://attacker@arxiv.org/... — netloc parsing oddities.
        assert not _is_pdf_url_safe("https://attacker@arxiv.org/pdf/x.pdf")

    def test_garbage_input_refused(self):
        from rag_system import _is_pdf_url_safe
        assert not _is_pdf_url_safe("not a url")
        assert not _is_pdf_url_safe("")


class TestDownloadArxivPdfRefusesUnsafe:
    """Even though the URL conversion logic produces an arxiv URL, an
    attacker-controlled paper_url that ends in a different host must
    still be refused after conversion."""

    @pytest.mark.asyncio
    async def test_non_arxiv_url_does_not_call_urlretrieve(self, monkeypatch, tmp_path):
        from rag_system import PDFDownloader

        downloader = PDFDownloader(cache_dir=str(tmp_path))
        called = {"hit": False}

        def fake_urlretrieve(url, dest):
            called["hit"] = True
            return None

        import urllib.request as ureq
        monkeypatch.setattr(ureq, "urlretrieve", fake_urlretrieve)

        # An attacker-supplied URL that contains 'arxiv' as a string but
        # lives on another host — must be refused.
        path = await downloader.download_arxiv_pdf("https://evil.example.com/arxiv.org/abs/123")
        assert path is None
        assert called["hit"] is False

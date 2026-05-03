"""Tests for src/assembled_core/certify/."""
from __future__ import annotations

import json
from datetime import datetime, timezone


from assembled_core.certify.schema import (
    EnvironmentFingerprint,
    InputFingerprint,
    OutputFingerprint,
    ReproducibilityCertificate,
)
from assembled_core.certify.generator import (
    file_sha256,
    object_sha256,
    get_git_info,
    get_environment_fingerprint,
    build_input_fingerprint,
    build_output_fingerprint,
    generate_certificate,
    save_certificate,
    verify_certificate,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemaDefaults:
    def test_env_fingerprint_defaults(self):
        e = EnvironmentFingerprint()
        assert isinstance(e.python_version, str)
        assert isinstance(e.package_hashes, dict)
        assert isinstance(e.random_seeds, dict)

    def test_input_fingerprint_defaults(self):
        inp = InputFingerprint()
        assert inp.data_file_hashes == {}
        assert inp.config_hash == ""

    def test_output_fingerprint_defaults(self):
        out = OutputFingerprint()
        assert out.equity_curve_hash == ""
        assert out.summary_metrics == {}

    def test_certificate_defaults(self):
        cert = ReproducibilityCertificate()
        assert cert.certificate_id == ""
        assert isinstance(cert.created_at, datetime)
        assert isinstance(cert.environment, EnvironmentFingerprint)


class TestCertificateSerialization:
    def _make_cert(self) -> ReproducibilityCertificate:
        return ReproducibilityCertificate(
            certificate_id="test-id-123",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            environment=EnvironmentFingerprint(
                python_version="3.11.9",
                platform="Linux",
                git_sha="abc123",
                git_dirty=False,
                package_hashes={"numpy": "1.24.0"},
                random_seeds={"python_random": 42},
            ),
            inputs=InputFingerprint(
                data_file_hashes={"data.parquet": "deadbeef"},
                config_hash="cafebabe",
                config_path="configs/strategy.yaml",
                model_hashes={},
            ),
            outputs=OutputFingerprint(
                equity_curve_hash="hash1",
                trades_hash="hash2",
                signals_hash="hash3",
                summary_metrics={"sharpe": 1.5, "max_drawdown": -0.12},
            ),
            notes="unit test",
        )

    def test_to_dict_roundtrip(self):
        cert = self._make_cert()
        d = cert.to_dict()
        assert d["certificate_id"] == "test-id-123"
        assert d["notes"] == "unit test"
        assert d["outputs"]["equity_curve_hash"] == "hash1"

    def test_from_dict_roundtrip(self):
        cert = self._make_cert()
        d = cert.to_dict()
        restored = ReproducibilityCertificate.from_dict(d)
        assert restored.certificate_id == cert.certificate_id
        assert restored.inputs.config_hash == cert.inputs.config_hash
        assert restored.outputs.summary_metrics["sharpe"] == 1.5
        assert restored.created_at == cert.created_at

    def test_to_json_valid(self):
        cert = self._make_cert()
        s = cert.to_json()
        parsed = json.loads(s)
        assert parsed["certificate_id"] == "test-id-123"


# ---------------------------------------------------------------------------
# Generator — hashing utilities
# ---------------------------------------------------------------------------

class TestFileSha256:
    def test_known_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world")
        h = file_sha256(f)
        assert len(h) == 64
        assert h == file_sha256(f)  # deterministic

    def test_missing_file_returns_not_found(self, tmp_path):
        h = file_sha256(tmp_path / "nonexistent.bin")
        assert h == "NOT_FOUND"

    def test_different_contents_differ(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert file_sha256(f1) != file_sha256(f2)


class TestObjectSha256:
    def test_dict_deterministic(self):
        d = {"b": 2, "a": 1}
        h1 = object_sha256(d)
        h2 = object_sha256(d)
        assert h1 == h2
        assert len(h1) == 64

    def test_key_order_invariant(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert object_sha256(d1) == object_sha256(d2)

    def test_different_values_differ(self):
        assert object_sha256({"x": 1}) != object_sha256({"x": 2})


# ---------------------------------------------------------------------------
# Generator — environment / git info
# ---------------------------------------------------------------------------

class TestGetGitInfo:
    def test_returns_tuple(self):
        sha, dirty = get_git_info()
        assert isinstance(sha, str)
        assert isinstance(dirty, bool)

    def test_sha_nonempty(self):
        sha, _ = get_git_info()
        assert len(sha) > 0


class TestGetEnvironmentFingerprint:
    def test_returns_env_fingerprint(self):
        fp = get_environment_fingerprint()
        assert isinstance(fp, EnvironmentFingerprint)

    def test_python_version_present(self):
        fp = get_environment_fingerprint()
        assert "3." in fp.python_version

    def test_platform_present(self):
        fp = get_environment_fingerprint()
        assert len(fp.platform) > 0


# ---------------------------------------------------------------------------
# Generator — build fingerprints
# ---------------------------------------------------------------------------

class TestBuildInputFingerprint:
    def test_existing_files_hashed(self, tmp_path):
        f = tmp_path / "prices.parquet"
        f.write_bytes(b"parquet-data")
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("strategy_id: test")
        inp = build_input_fingerprint([f], config_path=cfg)
        assert str(f) in inp.data_file_hashes
        assert inp.config_hash != ""
        assert inp.config_hash != "NOT_FOUND"

    def test_missing_data_file(self, tmp_path):
        inp = build_input_fingerprint([tmp_path / "nope.parquet"])
        hashes = list(inp.data_file_hashes.values())
        assert hashes[0] == "NOT_FOUND"

    def test_no_inputs(self):
        inp = build_input_fingerprint([])
        assert inp.data_file_hashes == {}
        assert inp.config_hash == ""


class TestBuildOutputFingerprint:
    def test_missing_dir_returns_empty(self, tmp_path):
        out = build_output_fingerprint(tmp_path / "nonexistent")
        assert out.equity_curve_hash == "NOT_FOUND"

    def test_with_artefacts(self, tmp_path):
        (tmp_path / "equity_curve.parquet").write_bytes(b"ec")
        (tmp_path / "trades.parquet").write_bytes(b"tr")
        (tmp_path / "signals.parquet").write_bytes(b"si")
        summary = {"sharpe": 1.2, "max_drawdown": -0.05}
        (tmp_path / "summary.json").write_text(json.dumps(summary))
        out = build_output_fingerprint(tmp_path)
        assert out.equity_curve_hash not in ("", "NOT_FOUND")
        assert out.trades_hash not in ("", "NOT_FOUND")
        assert out.summary_metrics["sharpe"] == 1.2


# ---------------------------------------------------------------------------
# Generator — generate_certificate / save / verify
# ---------------------------------------------------------------------------

class TestGenerateCertificate:
    def test_returns_certificate(self):
        cert = generate_certificate()
        assert isinstance(cert, ReproducibilityCertificate)

    def test_certificate_id_is_uuid(self):
        import uuid
        cert = generate_certificate()
        parsed = uuid.UUID(cert.certificate_id)
        assert parsed.version == 4

    def test_notes_propagated(self):
        cert = generate_certificate(notes="my note")
        assert cert.notes == "my note"

    def test_with_data_paths(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b\n1,2")
        cert = generate_certificate(data_paths=[f])
        assert str(f) in cert.inputs.data_file_hashes


class TestSaveAndVerify:
    def test_save_creates_file(self, tmp_path):
        cert = generate_certificate(notes="save test")
        path = tmp_path / "certificate.json"
        save_certificate(cert, path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["notes"] == "save test"

    def test_verify_matching_outputs(self, tmp_path):
        # Create output artefacts
        (tmp_path / "equity_curve.parquet").write_bytes(b"equity")
        (tmp_path / "trades.parquet").write_bytes(b"trades")
        (tmp_path / "signals.parquet").write_bytes(b"signals")

        cert = generate_certificate(output_dir=tmp_path)
        cert_path = tmp_path / "certificate.json"
        save_certificate(cert, cert_path)

        # Re-verify against same output dir → all should match
        results = verify_certificate(cert_path, tmp_path)
        assert results["equity_curve"] is True
        assert results["trades"] is True
        assert results["signals"] is True

    def test_verify_changed_outputs(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "equity_curve.parquet").write_bytes(b"original")
        (out_dir / "trades.parquet").write_bytes(b"original")
        (out_dir / "signals.parquet").write_bytes(b"original")

        cert = generate_certificate(output_dir=out_dir)
        cert_path = tmp_path / "cert.json"
        save_certificate(cert, cert_path)

        # Mutate an output file
        (out_dir / "equity_curve.parquet").write_bytes(b"changed!")

        results = verify_certificate(cert_path, out_dir)
        assert results["equity_curve"] is False
        assert results["trades"] is True

"""E-063 regression guard: every QuestDB connect must be time-bounded.

Background: ``_tc_execution`` Step 7.70 (QuestDB write-through) wraps
``tick_store.ping()`` in a broad ``except`` and was documented as "never
blocks the cycle". That guarantee was false — ``try/except`` bounds raises,
not wall-clock time. Against a black-holed host (firewall DROP instead of
REFUSE) the TCP connect hangs until the OS default and the ``except`` never
fires, stalling a synchronous trading-cycle step.

These tests pin the driver-level timeout that actually provides the bound.
Neither psycopg2 nor pg8000 is installed in this environment, so the driver
branches are exercised by monkeypatching ``_DRIVER``: the invariant under
test is which kwargs we HAND to the driver. The one test that checks the
driver's own contract (``test_real_driver_accepts_our_kwargs``) SKIPS here
and turns sharp at enablement, when a driver gets installed.
"""

from __future__ import annotations

import pytest

from src.assembled_core.data import tick_store

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("QUESTDB_CONNECT_TIMEOUT_S", raising=False)
    # _WARNED_ONCE is module state that is never cleared by design (warn-once
    # per process). Without this reset a future caplog assertion on the E-063
    # warning would pass alone and fail inside the suite.
    tick_store._WARNED_ONCE.clear()


def test_psycopg2_branch_sets_connect_timeout_and_keepalives(monkeypatch):
    monkeypatch.setattr(tick_store, "_DRIVER", "psycopg2")
    kwargs = tick_store._get_conn_kwargs()

    # libpq takes integer seconds and silently floors values below 2.
    assert isinstance(kwargs["connect_timeout"], int)
    assert kwargs["connect_timeout"] >= 2
    # Second hang class: an established connection whose peer vanishes.
    # The connect timeout does not cover that — keepalives do.
    assert kwargs["keepalives"] == 1
    for key in ("keepalives_idle", "keepalives_interval", "keepalives_count"):
        assert kwargs[key] > 0


def test_pg8000_branch_sets_connect_timeout(monkeypatch):
    monkeypatch.setattr(tick_store, "_DRIVER", "pg8000")
    kwargs = tick_store._get_conn_kwargs()

    assert kwargs["timeout"] == pytest.approx(tick_store._DEFAULT_CONNECT_TIMEOUT_S)
    assert kwargs["tcp_keepalive"] is True
    # pg8000 names it `timeout`, NOT `connect_timeout` — a wrong kwarg name
    # would raise TypeError inside _open_conn and silently disable the store.
    assert "connect_timeout" not in kwargs


@pytest.mark.parametrize("driver", ["psycopg2", "pg8000"])
def test_every_supported_driver_gets_a_bounded_connect(monkeypatch, driver):
    """The actual E-063 invariant: no supported driver may be called
    without a connect timeout, whatever the kwarg is named."""
    monkeypatch.setattr(tick_store, "_DRIVER", driver)
    kwargs = tick_store._get_conn_kwargs()
    timeout_keys = [k for k in kwargs if k in ("connect_timeout", "timeout")]
    assert timeout_keys, f"{driver} would connect unbounded"
    assert all(kwargs[k] > 0 for k in timeout_keys)


def test_env_override_is_honoured(monkeypatch):
    monkeypatch.setattr(tick_store, "_DRIVER", "pg8000")
    monkeypatch.setenv("QUESTDB_CONNECT_TIMEOUT_S", "7.5")
    assert tick_store._get_conn_kwargs()["timeout"] == pytest.approx(7.5)


@pytest.mark.parametrize(
    "raw",
    [
        "0",
        "-1",
        "-0.0",
        "abc",
        "",
        # The dangerous direction: "> 0" alone would accept all of these and
        # re-create the E-063 hang through a plausible operator typo.
        "600",
        "1e9",
        "inf",
        "nan",
    ],
)
def test_disabling_the_timeout_via_env_is_not_possible(monkeypatch, raw):
    """Fail-safe in BOTH directions: any value outside (0, 30] falls back to
    the default — the env var must never restore an unbounded connect."""
    monkeypatch.setattr(tick_store, "_DRIVER", "pg8000")
    monkeypatch.setenv("QUESTDB_CONNECT_TIMEOUT_S", raw)
    assert tick_store._get_connect_timeout_s() == pytest.approx(
        tick_store._DEFAULT_CONNECT_TIMEOUT_S
    )
    assert (
        0
        < tick_store._get_conn_kwargs()["timeout"]
        <= tick_store._MAX_CONNECT_TIMEOUT_S
    )


@pytest.mark.parametrize("raw", ["0.4", "0.5", "1", "1.9"])
def test_psycopg2_sub_two_second_values_are_floored_not_zeroed(monkeypatch, raw):
    """The most dangerous line of this change: int(round(0.4)) == 0, and
    libpq reads connect_timeout=0 as INFINITE — i.e. exactly the E-063 hang.
    The max(2, ...) floor is load-bearing and must stay pinned."""
    monkeypatch.setattr(tick_store, "_DRIVER", "psycopg2")
    monkeypatch.setenv("QUESTDB_CONNECT_TIMEOUT_S", raw)
    value = tick_store._get_conn_kwargs()["connect_timeout"]
    assert value != 0, "connect_timeout=0 means UNLIMITED in libpq"
    assert value == 2


def test_var_keyword_driver_is_not_filtered(monkeypatch):
    """A driver whose signature is (user, **kwargs) accepts names that are
    NOT in its parameter list. Filtering against that list would drop
    host/port/password and connect to the WRONG database silently, or
    falsely report `timeout` as unsupported."""
    monkeypatch.setattr(tick_store, "_DRIVER", "pg8000")

    def _driver(user=None, **kwargs):
        return None

    monkeypatch.setattr(tick_store, "_connect_fn", _driver)
    kwargs = tick_store._validate_conn_kwargs(tick_store._get_conn_kwargs())
    assert kwargs is not None, "**kwargs driver must not be declared unusable"
    assert kwargs["timeout"] > 0
    # No silent connection-target loss:
    assert kwargs["host"] and kwargs["port"] and kwargs["database"]


def test_named_and_var_keyword_driver_keeps_all_kwargs(monkeypatch):
    monkeypatch.setattr(tick_store, "_DRIVER", "pg8000")

    def _driver(user=None, timeout=None, **kwargs):
        return None

    monkeypatch.setattr(tick_store, "_connect_fn", _driver)
    kwargs = tick_store._validate_conn_kwargs(tick_store._get_conn_kwargs())
    assert kwargs is not None
    assert kwargs["host"] and kwargs["database"]


def test_unknown_driver_without_timeout_mapping_is_refused(monkeypatch):
    """Structural invariant, not per-branch discipline: adding a third
    driver (psycopg3) without extending _get_conn_kwargs must NOT silently
    restore the unbounded connect."""
    monkeypatch.setattr(tick_store, "_DRIVER", "psycopg3")
    kwargs = tick_store._get_conn_kwargs()
    assert "connect_timeout" not in kwargs and "timeout" not in kwargs
    assert tick_store._validate_conn_kwargs(kwargs) is None
    monkeypatch.setattr(tick_store, "_connect_fn", lambda **kw: object())
    assert tick_store._open_conn() is None


def test_psycopg2_keepalive_rejection_falls_back_to_connect_timeout(monkeypatch):
    """Stubbed make_dsn so the psycopg2 branch runs without the driver
    installed: keepalives rejected -> keep going with connect_timeout;
    connect_timeout rejected -> refuse to connect (fail-closed)."""
    import sys
    import types

    def _make_stub(reject: set[str]):
        mod = types.ModuleType("psycopg2")
        ext = types.ModuleType("psycopg2.extensions")

        def make_dsn(**kw):
            bad = reject & set(kw)
            if bad:
                raise ValueError(f"invalid connection option {sorted(bad)[0]!r}")
            return "dsn"

        ext.make_dsn = make_dsn  # type: ignore[attr-defined]
        mod.extensions = ext  # type: ignore[attr-defined]
        return mod, ext

    monkeypatch.setattr(tick_store, "_DRIVER", "psycopg2")

    mod, ext = _make_stub({"keepalives", "keepalives_idle"})
    monkeypatch.setitem(sys.modules, "psycopg2", mod)
    monkeypatch.setitem(sys.modules, "psycopg2.extensions", ext)
    kwargs = tick_store._validate_conn_kwargs(tick_store._get_conn_kwargs())
    assert kwargs is not None
    assert kwargs["connect_timeout"] >= 2
    assert not any(k.startswith("keepalives") for k in kwargs)

    mod2, ext2 = _make_stub({"connect_timeout"})
    monkeypatch.setitem(sys.modules, "psycopg2", mod2)
    monkeypatch.setitem(sys.modules, "psycopg2.extensions", ext2)
    assert tick_store._validate_conn_kwargs(tick_store._get_conn_kwargs()) is None


def test_no_driver_means_no_connect_attempt(monkeypatch):
    """Without a driver _open_conn must return None without touching the
    network (nothing to time out)."""
    monkeypatch.setattr(tick_store, "_connect_fn", None)
    assert tick_store._open_conn() is None
    assert tick_store.ping() is False


def test_unusable_driver_kwargs_disable_the_store_instead_of_connecting(monkeypatch):
    """E-063 fail-CLOSED: if no time-bounded call can be built, we must NOT
    connect at all — an unbounded connect is worse than no tick store."""
    monkeypatch.setattr(tick_store, "_DRIVER", "pg8000")

    def _driver_without_timeout(host=None, port=None, user=None, password=None):
        raise AssertionError("must not be called")

    monkeypatch.setattr(tick_store, "_connect_fn", _driver_without_timeout)
    assert tick_store._validate_conn_kwargs(tick_store._get_conn_kwargs()) is None
    assert tick_store._open_conn() is None


def test_driver_signature_drift_drops_unknown_kwargs(monkeypatch):
    """A driver with an EXPLICIT signature (no **kwargs) that still has
    `timeout` but dropped another kwarg must not die with a TypeError —
    only unknown keys are filtered. Filtering is safe here precisely
    because the signature is exhaustive (see test_var_keyword_driver_*)."""
    monkeypatch.setattr(tick_store, "_DRIVER", "pg8000")

    def _driver(host=None, port=None, user=None, password=None, timeout=None):
        return None

    monkeypatch.setattr(tick_store, "_connect_fn", _driver)
    kwargs = tick_store._validate_conn_kwargs(tick_store._get_conn_kwargs())
    assert kwargs is not None
    assert kwargs["timeout"] > 0
    assert kwargs["host"] and kwargs["port"]
    assert "database" not in kwargs and "tcp_keepalive" not in kwargs


@pytest.mark.parametrize("driver_mod", ["psycopg2", "pg8000"])
def test_real_driver_accepts_our_kwargs(driver_mod):
    """Contract test against the REAL driver. Skips while the driver is not
    installed (the state in this repo today) and turns sharp exactly at
    enablement — which is when the kwarg names start to matter.

    Verified manually on 2026-08-01 against psycopg2-binary 2.9.12 and
    pg8000 1.31.5.
    """
    pytest.importorskip(driver_mod)
    import importlib

    driver = importlib.import_module(driver_mod)

    if driver_mod == "psycopg2":
        from psycopg2 import extensions

        kwargs = _kwargs_for(driver_mod)
        extensions.make_dsn(**kwargs)  # raises if any option is unknown
    else:
        import inspect as _inspect

        params = set(_inspect.signature(driver.connect).parameters)
        assert set(_kwargs_for(driver_mod)).issubset(params)


def _kwargs_for(driver_name: str) -> dict:
    import unittest.mock as _mock

    with _mock.patch.object(tick_store, "_DRIVER", driver_name):
        return tick_store._get_conn_kwargs()


def test_connect_receives_the_timeout_kwarg(monkeypatch):
    """End-to-end for the wiring: whatever _get_conn_kwargs produces is what
    the driver callable actually receives."""
    seen: dict = {}

    def _fake_connect(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(tick_store, "_DRIVER", "psycopg2")
    monkeypatch.setattr(tick_store, "_connect_fn", _fake_connect)
    assert tick_store._open_conn() is not None
    assert seen["connect_timeout"] >= 2

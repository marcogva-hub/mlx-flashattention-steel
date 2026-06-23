"""Volet E3 / CX-07 — MFA_NO_PADDING is load-time-only and NOT invalidatable.

The value is frozen at first read (function-local static in mfa_env.hpp::no_padding)
because MFA_NO_PADDING is absent from the shader-cache KernelKey — a mid-process
toggle would otherwise return a stale-padding kernel. This locks the *documented*
semantics (fix-path b): honored when set before first read; inert thereafter, even
across _invalidate_env_config(). Subprocesses control first-read timing.
"""
import subprocess
import sys
import textwrap


def _run(code):
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"subprocess failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout.strip()


def test_no_padding_honored_when_set_before_first_read():
    out = _run("""
        import os
        os.environ["MFA_NO_PADDING"] = "1"   # set BEFORE import/first read
        import mlx_mfa._ext as e
        print(e._mfa_no_padding_frozen())
    """)
    assert out == "True", f"set-before-first-read should be honored, got {out}"


def test_no_padding_inert_after_first_read_and_invalidate():
    # Freeze False (env unset), then set it + invalidate → must stay False (CX-07:
    # load-time-only; _invalidate_env_config does NOT reset it).
    out = _run("""
        import os
        os.environ.pop("MFA_NO_PADDING", None)
        import mlx_mfa, mlx_mfa._ext as e
        first = e._mfa_no_padding_frozen()      # freezes False
        os.environ["MFA_NO_PADDING"] = "1"      # change AFTER first read
        mlx_mfa._invalidate_env_config()        # documented: does NOT reset this var
        second = e._mfa_no_padding_frozen()
        print(f"{first},{second}")
    """)
    assert out == "False,False", (
        f"MFA_NO_PADDING must be inert after first read + invalidate (load-time-only), "
        f"got {out}")

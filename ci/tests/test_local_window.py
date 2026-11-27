"""Inkling release-tier leg: it runs in its own gpu-window but starts its own container (no homelab unit).

Order is the contract: acquire -> stop prod -> docker run -> health ... docker rm -f -> start prod -> release.
Cleanup must run on every exit path (failed start, exception in a probe, SIGTERM) and only once.
"""
import re
import signal
import sys
import unittest
from pathlib import Path

CI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI / "accept"))
import accept  # noqa: E402

PROD = "docker-vllm7.service"


class Fake:
    """stands in for subprocess/ssh/http: records every call in order"""

    def __init__(self, fail_on=None, healthy=True):
        self.calls, self.fail_on, self.healthy = [], fail_on, healthy

    def run(self, argv, **kw):
        self.calls.append(" ".join(argv))
        if self.fail_on and self.fail_on in " ".join(argv):
            raise RuntimeError(f"boom: {' '.join(argv)}")
        if argv[:2] == ["docker", "logs"]:
            return "GPU KV cache size: 53,608 tokens\n"
        return ""

    def ssh(self, *args, **kw):
        self.calls.append("ssh " + " ".join(args))
        return '{"health": 200}'


def window(fake, **kw):
    w = accept.LocalDockerWindow("gpuwin@x", None, "o", 600, run=fake.run, ssh=fake.ssh,
                                 health=lambda url: 200 if fake.healthy else 0,
                                 sleep=lambda s: None, prod_service=PROD, install_signals=False, **kw)
    return w


def idx(calls, needle):
    """last match: the stale-container `docker rm -f` before `docker run` is not the cleanup one"""
    return max(i for i, c in enumerate(calls) if needle in c)


class Order(unittest.TestCase):
    def test_happy_path_sequence(self):
        f = Fake()
        w = window(f)
        w.acquire()
        res = w.start("vllm-sly-radiance:ci-1-abc", "unused")
        self.assertEqual(res["kv_tokens"], 53608)
        w.release()
        c = f.calls
        order = [idx(c, "ssh acquire"), idx(c, f"systemctl stop {PROD}"), idx(c, "docker run"),
                 idx(c, "docker rm -f"), idx(c, f"systemctl start {PROD}"), idx(c, "ssh release")]
        self.assertEqual(order, sorted(order), c)
        self.assertEqual(len(set(order)), 6)

    def test_start_failure_still_restores_prod(self):
        f = Fake(fail_on="docker run")
        w = window(f)
        w.acquire()
        with self.assertRaises(RuntimeError):
            w.start("img:t", "x")
        w.release()
        self.assertIn(f"systemctl start {PROD}", " | ".join(f.calls))
        self.assertIn("docker rm -f", " | ".join(f.calls))

    def test_failed_stop_still_tries_to_restore_prod(self):
        f = Fake(fail_on=f"systemctl stop {PROD}")
        w = window(f)
        w.acquire()
        with self.assertRaises(RuntimeError):
            w.start("img:t", "x")
        w.release()
        self.assertIn(f"systemctl start {PROD}", " | ".join(f.calls))

    def test_health_timeout_removes_container_and_restores(self):
        f = Fake(healthy=False)
        w = window(f, health_wait_s=0)
        w.acquire()
        with self.assertRaises(RuntimeError):
            w.start("img:t", "x")
        w.release()
        self.assertLess(idx(f.calls, "docker rm -f"), idx(f.calls, f"systemctl start {PROD}"))
        self.assertLess(idx(f.calls, "docker run"), idx(f.calls, "docker rm -f"))

    def test_cleanup_is_idempotent_and_prod_is_not_started_if_never_stopped(self):
        f = Fake()
        w = window(f)
        w.acquire()
        w.release()  # nothing was started
        w.cleanup()
        self.assertNotIn(f"systemctl start {PROD}", " | ".join(f.calls))
        f2 = Fake()
        w2 = window(f2)
        w2.acquire()
        w2.start("img:t", "x")
        w2.cleanup()
        w2.cleanup()
        self.assertEqual(sum(1 for c in f2.calls if c == f"sudo -n systemctl start {PROD}"), 1)

    def test_sigterm_handler_turns_into_exit_so_finally_blocks_run(self):
        old = signal.getsignal(signal.SIGTERM)
        try:
            accept.LocalDockerWindow.install_signal_handlers()
            with self.assertRaises(SystemExit):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        finally:
            signal.signal(signal.SIGTERM, old)


class ReleaseErrors(unittest.TestCase):
    def test_failed_prod_restart_is_reported_but_the_helper_release_still_runs(self):
        f = Fake(fail_on=f"systemctl start {PROD}")
        w = window(f)
        w.acquire()
        w.start("img:t", "x")
        with self.assertRaises(RuntimeError):
            w.release()
        self.assertIn("ssh release", f.calls)


class Argv(unittest.TestCase):
    def argv(self, **kw):
        return accept.inkling_docker_argv("img:t", "cname", 8001, "/m", "/d", "/c", **kw)

    def test_one_env_per_dash_e(self):
        a = self.argv()
        for i, x in enumerate(a):
            if x == "-e":
                self.assertRegex(a[i + 1], r"^[A-Z_0-9]+=[^ ]*$")

    def test_release_arm_args(self):
        j = " ".join(self.argv())
        for needle in ("--served-model-name inkling2-mini", "--max-model-len 32768", "--max-num-seqs 8",
                       "--reasoning-parser inkling", "--tool-call-parser inkling", "--no-async-scheduling",
                       "-p 8001:8000", "-v /m:/model:ro", "-v /d:/models/dflash2:ro"):
            self.assertIn(needle, j)
        # the gate must test the image default: no model env is forced
        self.assertNotIn("RADIANCE_INKLING2", j)

    def test_matches_the_backtest_script(self):
        """quick_backtest.sh start_inkling is the hand-run twin: its flags must not drift from the CI leg"""
        sh = (CI / "quick_backtest.sh").read_text()
        for tok in re.findall(r"--[a-z-]+(?: \S+)?", " ".join(self.argv()[self.argv().index("/model"):])):
            self.assertIn(tok.split()[0], sh, tok)
        for kv in [x for x in self.argv() if re.match(r"^RADIANCE_\w+=1$", x)]:
            self.assertIn(kv, sh)

    def test_extra_env(self):
        self.assertIn("RADIANCE_DFLASH_CONV_W4=1", self.argv(extra_env=["RADIANCE_DFLASH_CONV_W4=1"]))


if __name__ == "__main__":
    unittest.main()

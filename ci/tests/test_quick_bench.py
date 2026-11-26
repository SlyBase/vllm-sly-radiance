"""Regression gates (ci/quick_bench.py): pure checkers, driven by real excerpts of host-51 runs.

Fixtures (ci/tests/fixtures/gates): QP2 = 1.13.4 (good Qwen), QC2 = eager Qwen (bad), QCL = fixed Qwen,
inkling-good = ci-262 Inkling release arm, bb-conc-c* = Inkling BetterBench concurrency 1/2/4/8,
bb-decode-qwen = QP2 BetterBench decode, greedy-eager = the QC2 greedy output.
"""
import json
import sys
import unittest
from pathlib import Path

CI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI))
import quick_bench as qb  # noqa: E402

FX = CI / "tests" / "fixtures" / "gates"
REF = CI / "accept" / "references"


def fx(name):
    return (FX / name).read_text()


class HookProof(unittest.TestCase):
    def test_qwen_good_and_fixed_pass(self):
        self.assertEqual(qb.hook_problems(fx("qwen-good-1.13.4.log"), "qwen"), [])
        self.assertEqual(qb.hook_problems(fx("qwen-fixed.log"), "qwen"), [])

    def test_qwen_eager_fails_on_both_signals(self):
        p = " ".join(qb.hook_problems(fx("qwen-eager-bad.log"), "qwen"))
        self.assertIn("add_rms_quant active", p)
        self.assertIn("CompilationMode", p)

    def test_inkling_good_passes(self):
        self.assertEqual(qb.hook_problems(fx("inkling-good.log"), "inkling"), [])

    def test_inkling_int4_drafter_fails(self):
        log = fx("inkling-good.log").replace("conv projections int4 0", "conv projections int4 5")
        self.assertTrue(qb.hook_problems(log, "inkling"))

    def test_inkling_missing_line_and_short_cudagraph_sizes_fail(self):
        no_cap = fx("inkling-good.log").replace("Capturing model for DFlash2 speculator", "")
        self.assertIn("Capturing model", " ".join(qb.hook_problems(no_cap, "inkling")))
        short = fx("inkling-good.log").replace("16, 24, 32, 40, 48, 56, 64", "16")
        self.assertIn("cudagraph", " ".join(qb.hook_problems(short, "inkling")))


class Greedy(unittest.TestCase):
    ref = json.loads((REF / "quick-qwen-greedy.json").read_text())

    def test_identical_passes(self):
        self.assertEqual(qb.greedy_problems(self.ref, self.ref, {}), [])

    def test_eager_output_differs(self):
        cur = json.loads(fx("greedy-eager.json"))
        self.assertEqual(len(qb.greedy_problems(cur, self.ref, {})), 4)

    def test_allow_list_needs_a_reason(self):
        cur = json.loads(fx("greedy-eager.json"))
        allow = {str(i): "new torch changes reduction order" for i in range(4)}
        self.assertEqual(qb.greedy_problems(cur, self.ref, allow), [])
        self.assertEqual(len(qb.greedy_problems(cur, self.ref, {"0": ""})), 4)

    def test_wrong_count_fails(self):
        self.assertTrue(qb.greedy_problems(self.ref[:3], self.ref, {}))


class Concurrency(unittest.TestCase):
    def agg(self):
        out = {}
        for c in (1, 2, 4, 8):
            for e in json.loads(fx(f"bb-conc-c{c}.json"))["concurrency"]:
                out[e["level"]] = e["aggregate_tps"]
        return out

    def test_real_inkling_run_passes(self):
        self.assertEqual(qb.concurrency_problems(self.agg()), [])

    def test_two_stream_collapse_fails(self):
        a = self.agg()
        a[2] = a[1] * 1.1
        self.assertTrue(qb.concurrency_problems(a))

    def test_non_monotonic_fails(self):
        a = self.agg()
        a[4] = a[2] * 0.9
        self.assertTrue(qb.concurrency_problems(a))

    def test_fast_mode_without_level_2_only_checks_monotony(self):
        self.assertEqual(qb.concurrency_problems({1: 100.0, 8: 300.0}), [])
        self.assertTrue(qb.concurrency_problems({1: 100.0, 8: 90.0}))


class DrafterFloor(unittest.TestCase):
    def test_tok_per_update_from_betterbench_json(self):
        tpu = qb.category_tok_per_update(json.loads(fx("bb-decode-qwen.json")))
        self.assertAlmostEqual(tpu["code"], 5.5, delta=0.05)

    def test_floor(self):
        self.assertEqual(qb.floor_problems({"code": 5.5}, 4.8), [])
        self.assertTrue(qb.floor_problems({"code": 3.65}, 4.8))  # the int4 drafter (D1)
        self.assertTrue(qb.floor_problems({}, 4.8))  # nothing measured is not a pass


class Verdict(unittest.TestCase):
    ref = {"decode_tps": {"code": 200.0, "prose": 100.0}, "prefill_pps": 4000.0,
           "tok_per_update": {"code": 5.0}, "c2_over_c1_min": 1.2, "c4_over_c1_min": 1.4}
    thr = {"decode_pct": 2.0, "prefill_pct": 3.0}
    good = {"decode_tps": {"code": 199.0, "prose": 99.0}, "prefill_pps": 3950.0,
            "tok_per_update": {"code": 5.1}, "c2_over_c1": 1.31, "c4_over_c1": 1.57}

    def bad(self, **kw):
        return {**self.good, **kw}

    def names(self, m):
        return [n for n, ok, _ in qb.verdict(m, self.ref, self.thr) if not ok]

    def test_good_passes(self):
        self.assertEqual(self.names(self.good), [])

    def test_each_regression_is_flagged(self):
        self.assertEqual(self.names(self.bad(decode_tps={"code": 190.0, "prose": 99.0})), ["decode code"])
        self.assertEqual(self.names(self.bad(prefill_pps=3000.0)), ["prefill"])
        self.assertEqual(self.names(self.bad(tok_per_update={"code": 3.65})), ["tok/update code"])
        self.assertEqual(self.names(self.bad(c4_over_c1=1.1)), ["c4/c1"])
        self.assertEqual(self.names(self.bad(c2_over_c1=1.05)), ["c2/c1"])  # the 2-stream collapse

    def test_uncalibrated_reference_entries_are_skipped_not_failed(self):
        ref = {"decode_tps": {"code": None}, "prefill_pps": None, "tok_per_update": {}, "c2_over_c1_min": None, "c4_over_c1_min": None}
        self.assertTrue(all(ok for _, ok, _ in qb.verdict(self.good, ref, self.thr)))


class Metrics(unittest.TestCase):
    def test_tok_per_update_from_prom_delta(self):
        a = {"vllm:spec_decode_num_drafts_total": 100.0, "vllm:spec_decode_num_accepted_tokens_total": 400.0}
        b = {"vllm:spec_decode_num_drafts_total": 200.0, "vllm:spec_decode_num_accepted_tokens_total": 900.0}
        self.assertAlmostEqual(qb.tok_per_update(a, b), 6.0)
        self.assertIsNone(qb.tok_per_update(a, a))

    def test_parse_prom(self):
        txt = 'vllm:spec_decode_num_drafts_total{engine="0"} 12.0\n# c\nother 1\n'
        self.assertEqual(qb.parse_prom(txt), {"vllm:spec_decode_num_drafts_total": 12.0})


class Gsm8k(unittest.TestCase):
    def test_is_correct(self):
        self.assertTrue(qb.gsm8k_correct("so\n#### 1,366", "1366"))
        self.assertTrue(qb.gsm8k_correct("the answer is 18.", "18"))  # fallback: last number
        self.assertFalse(qb.gsm8k_correct("#### 17", "18"))
        self.assertFalse(qb.gsm8k_correct("no number here", "18"))

    def test_items_file_has_50(self):
        self.assertEqual(len(qb.load_gsm8k()), 50)

    def test_floor_is_two_items_below_the_reference(self):
        ref = {"gsm8k_correct": 40}
        ok = lambda n: [x for x in qb.verdict({"gsm8k_correct": n}, ref, {"decode_pct": 2, "prefill_pct": 3}) if not x[1]]  # noqa: E731
        self.assertEqual(ok(38), [])
        self.assertEqual([n for n, *_ in ok(37)], ["gsm8k"])


class ReleaseGuard(unittest.TestCase):
    full = {"decode_tps": {"prose": 1.0, "code": 2.0}, "prefill_pps": 3000.0, "tok_per_update": {"code": 5.0},
            "c2_over_c1_min": 1.2, "c4_over_c1_min": 1.4, "c8_over_c1_min": 1.6, "gsm8k_correct": 40}

    def test_complete_reference_passes(self):
        self.assertEqual(qb.unrecorded_problems(self.full, greedy_present=True), [])

    def test_each_missing_piece_is_named_and_the_hint_is_there(self):
        for key in ("decode_tps", "prefill_pps", "gsm8k_correct", "c8_over_c1_min", "tok_per_update"):
            ref = {k: v for k, v in self.full.items() if k != key}
            msg = " ".join(qb.unrecorded_problems(ref, greedy_present=True))
            self.assertIn(key, msg)
            self.assertIn("references not recorded, run ci/quick_backtest.sh RECORD=1", msg)
        ref = {**self.full, "decode_tps": {"prose": 1.0, "code": None}}
        self.assertTrue(qb.unrecorded_problems(ref, greedy_present=True))
        self.assertIn("greedy", " ".join(qb.unrecorded_problems(self.full, greedy_present=False)))

    def test_committed_seed_references_fail_closed(self):
        for prof in ("qwen", "inkling"):
            ref = json.loads((REF / f"quick-{prof}.json").read_text())
            self.assertTrue(qb.unrecorded_problems(ref, greedy_present=True), prof)


class ReleaseTierProfiles(unittest.TestCase):
    def test_both_profiles_and_their_files(self):
        self.assertEqual(qb.RELEASE_PROFILES, ("vllm7-mxfp4", "inkling"))
        for p in qb.RELEASE_PROFILES:
            prof = json.loads((CI / "accept" / "profiles" / f"{p}.json").read_text())
            self.assertIn(prof["quick_profile"], qb.HOOKS)
            self.assertTrue(prof["service"].endswith(".service"))


class ReleaseGate(unittest.TestCase):
    ctx = ["release-tier/vllm7-mxfp4", "release-tier/inkling"]

    def test_both_green(self):
        st = {c: "success" for c in self.ctx}
        self.assertEqual(qb.release_gate_problems(st), [])

    def test_missing_or_red_blocks(self):
        self.assertEqual(len(qb.release_gate_problems({"release-tier/vllm7-mxfp4": "success"})), 1)
        self.assertEqual(len(qb.release_gate_problems({self.ctx[0]: "success", self.ctx[1]: "failure"})), 1)
        self.assertEqual(len(qb.release_gate_problems({})), 2)
        # an unrelated green status is not a release tier
        self.assertEqual(len(qb.release_gate_problems({"ci": "success"})), 2)


class AcceptWiring(unittest.TestCase):
    """accept.compare() turns the gates into hard checks."""

    def checks(self, **results):
        sys.path.insert(0, str(CI / "accept"))
        import accept
        profile = json.loads((CI / "accept" / "profiles" / "vllm7-mxfp4.json").read_text())
        out = []
        accept.gate_checks(results, profile, lambda name, ok, actual, expected, hard=True: out.append((name, ok)))
        return dict(out)

    def test_hooks_greedy_concurrency_floor(self):
        ref = json.loads((REF / "quick-qwen-greedy.json").read_text())
        c = self.checks(hooks=[], greedy=ref, betterbench={"aggregate_tps": {"1": 100, "2": 170, "4": 250, "8": 300},
                                                          "categories_tokens_per_update": {"code": 5.5}})
        self.assertTrue(all(c.values()) and len(c) == 4, c)
        c = self.checks(hooks=["missing: x"], greedy=json.loads(fx("greedy-eager.json")),
                        betterbench={"aggregate_tps": {"1": 100, "2": 110}, "categories_tokens_per_update": {"code": 3.6}})
        self.assertFalse(any(c.values()), c)


class AcceptReleaseMode(AcceptWiring):
    def test_unrecorded_reference_is_a_hard_failure_in_the_report(self):
        q = {"rows": [], "unrecorded": [qb.RECORD_HINT + " (missing: decode_tps)"], "failures": ["x"]}
        c = self.checks(quick=q)
        self.assertFalse(c["references recorded"])
        self.assertTrue(self.checks(quick={"rows": [("decode code", True, "ok")], "unrecorded": [], "failures": []})["references recorded"])


class HarnessLint(unittest.TestCase):
    def test_flags_two_assignments_in_one_dash_e(self):
        self.assertTrue(qb.multi_env_problems('docker run -e "A=1 B=2" img'))
        self.assertTrue(qb.multi_env_problems("docker run -e A=1 B=2 img"))
        self.assertEqual(qb.multi_env_problems('docker run -e A=1 -e B=2 -e "C=x y" img'), [])
        # a closing quote ends the -e value: the next KEY=VAL belongs to the shell, not to docker
        self.assertEqual(qb.multi_env_problems('X="$a -e B=1 -e C=2" IMAGE=$I run.sh'), [])


class FakeServer(unittest.TestCase):
    """The live probes against a tiny local server: parsing, wiring and the end-to-end verdict (no GPU)."""

    @classmethod
    def setUpClass(cls):
        import http.server
        import threading
        state = cls.state = {"drafts": 0, "acc": 0}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = (f'vllm:spec_decode_num_drafts_total{{e="0"}} {state["drafts"]}\n'
                        f'vllm:spec_decode_num_accepted_tokens_total{{e="0"}} {state["acc"]}\n').encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                n = req.get("max_tokens", 8)
                state["drafts"] += n // 5
                state["acc"] += 4 * (n // 5)
                self.send_response(200)
                self.end_headers()
                if req.get("stream"):
                    for i in range(4):
                        d = {"text": "word " * 8} if "prompt" in req else {"delta": {"content": "word " * 8}}
                        self.wfile.write(b"data: " + json.dumps({"choices": [d]}).encode() + b"\n\n")
                        __import__("time").sleep(0.01)
                    u = {"usage": {"completion_tokens": n}, "choices": []}
                    self.wfile.write(b"data: " + json.dumps(u).encode() + b"\n\ndata: [DONE]\n\n")
                else:
                    r = {"choices": [{"finish_reason": "stop", "message": {"content": "answer " + req["messages"][0]["content"][:5] + "\\n#### 7"}}],
                         "usage": {"completion_tokens": n}}
                    self.wfile.write(json.dumps(r).encode())

        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_probes_measure_something_sane(self):
        tps, tpu = qb.decode_probe(self.base, "m", reps=2, tokens=100)
        self.assertGreater(tps["code"], 0)
        self.assertAlmostEqual(tpu["code"], 5.0, delta=0.5)  # 4 accepted per draft + 1
        self.assertGreater(qb.prefill_probe(self.base, "m", n=64, reps=1), 0)
        agg = qb.concurrency_probe(self.base, "m", requests=4, tokens=16)
        self.assertEqual(sorted(agg), [1, 2, 4])
        self.assertEqual(len(qb.greedy_run(self.base, "m")), 4)

    def test_run_quick_record_then_pass_then_hook_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ref_dir = Path(d)
            log = fx("qwen-good-1.13.4.log")
            report, fails = qb.run_quick("qwen", self.base, "m", log, ref_dir)
            self.assertEqual(fails, [])  # no reference yet: nothing to compare, hooks fine
            qb.record("qwen", report, ref_dir)
            report, fails = qb.run_quick("qwen", self.base, "m", log, ref_dir)
            self.assertEqual([f for f in fails if f.startswith("greedy")], [])
            _, fails = qb.run_quick("qwen", self.base, "m", fx("qwen-eager-bad.log"), ref_dir)
            self.assertTrue(any(f.startswith("hook proof") for f in fails))


class FakeRelease(FakeServer):
    """release tier against the fake server"""
    test_probes_measure_something_sane = None
    test_run_quick_record_then_pass_then_hook_failure = None

    def test_release_fails_closed_on_unrecorded_references_without_probing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            before = dict(self.state)
            report, fails = qb.run_quick("qwen", self.base, "m", fx("qwen-good-1.13.4.log"), Path(d), release=True)
            self.assertIn("references not recorded, run ci/quick_backtest.sh RECORD=1", " ".join(fails))
            self.assertEqual(self.state, before)  # no probe was sent
            self.assertTrue(report["unrecorded"])

    def test_release_record_then_pass_with_gsm8k_and_c8(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ref_dir = Path(d)
            log = fx("qwen-good-1.13.4.log")
            report, _ = qb.run_quick("qwen", self.base, "m", log, ref_dir, release=True, gsm8k_n=3, recording=True)
            self.assertEqual(sorted(report["measured"]["concurrency"]), [1, 2, 4, 8])
            self.assertIn(report["measured"]["gsm8k_correct"], range(4))  # the fake answers "#### 7" to everything
            qb.record("qwen", report, ref_dir)
            ref = json.loads((ref_dir / "quick-qwen.json").read_text())
            self.assertEqual(qb.unrecorded_problems(ref, greedy_present=True), [])


if __name__ == "__main__":
    unittest.main()

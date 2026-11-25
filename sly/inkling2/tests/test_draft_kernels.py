"""DSpark draft kernels of radiance_inkling2_draft against float64 torch references.

The draft's launches (~10 torch ops per RMSNorm, ~10 launches and a 103 MB hipBLASLt GEMM per Markov
step) are what the module folds: every kernel here does the same math as the op sequence it replaces,
fp32 summation order aside. CPU, Triton interpreter: weights are fp32 (bf16-representable) since the
interpreter's bf16 tl.dot is wrong; the kernels take their dot dtype from the weight dtype.
"""
import pytest
import torch


def bf16(x):
    return x.bfloat16().double()


# ---------------------------------------------------------------- RMSNorm


def ref_norm(v, w, eps):
    v = v.double()
    return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * w.double()).bfloat16()


@pytest.mark.parametrize("rows,n", [(5, 64), (3, 2304), (7, 128), (1, 100)])
def test_rmsnorm_without_residual(rows, n):
    from radiance_inkling2_draft import rmsnorm

    g = torch.Generator().manual_seed(1)
    x = torch.randn(rows, n, generator=g).bfloat16()
    w = (torch.rand(n, generator=g) + 0.5).bfloat16()
    x0 = x.clone()
    out = rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(out.float(), ref_norm(x.float(), w, 1e-6).float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x, x0, rtol=0, atol=0)  # the input of the plain norm is untouched


@pytest.mark.parametrize("rows,n", [(5, 64), (3, 2304)])
def test_rmsnorm_with_residual_matches_the_flashinfer_contract(rows, n):
    """residual += x (rounded to bf16 once), x = norm(UNROUNDED fp32 sum), both in place."""
    from radiance_inkling2_draft import rmsnorm

    g = torch.Generator().manual_seed(2)
    x = torch.randn(rows, n, generator=g).bfloat16()
    r = torch.randn(rows, n, generator=g).bfloat16()
    w = (torch.rand(n, generator=g) + 0.5).bfloat16()
    total = x.float() + r.float()
    xs, rs = x.clone(), r.clone()
    out = rmsnorm(xs, w, 1e-6, rs)
    torch.testing.assert_close(rs, total.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(out.float(), ref_norm(total, w, 1e-6).float(), atol=2e-2, rtol=2e-2)
    assert out.data_ptr() == xs.data_ptr()


def test_rmsnorm_rows_with_a_stride():
    from radiance_inkling2_draft import rmsnorm

    base = torch.randn(4, 96).bfloat16()
    x = base[:, :64]
    w = torch.ones(64).bfloat16()
    torch.testing.assert_close(rmsnorm(x, w, 1e-6).float(), ref_norm(x.float(), w, 1e-6).float(),
                               atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------- Markov step


def ref_step(state, prev, context, w1, w):
    r = state.shape[1]
    x = torch.cat([state.double(), w1[prev].double()], -1)
    raw = context.double() + x @ w.double().T
    gate, cand, out = bf16(raw[:, :r]), bf16(raw[:, r:2 * r]), bf16(raw[:, 2 * r:])
    gate = bf16(torch.sigmoid(gate))
    cand = bf16(torch.tanh(cand))
    keep = bf16(gate * state.double())
    update = bf16(bf16(1.0 - gate) * cand)
    return bf16(keep + update), bf16(torch.tanh(out))


@pytest.mark.parametrize("rows,r", [(1, 32), (3, 64), (8, 32)])
def test_markov_step_matches_the_op_sequence(rows, r):
    from radiance_inkling2_draft import markov_step

    g = torch.Generator().manual_seed(3)
    V = 50
    state = torch.tanh(torch.randn(rows, r, generator=g)).bfloat16()
    prev = torch.randint(0, V, (rows,), generator=g)
    context = torch.randn(rows, 3 * r, generator=g)
    w1 = torch.randn(V, r, generator=g).bfloat16().float()
    # rnn_proj.weight[:, :2r]: a column slice of the wider matrix (row stride > 2r)
    wide = (torch.randn(3 * r, 2 * r + 24, generator=g) * 0.2).bfloat16().float()
    w = wide[:, :2 * r]
    new_state, latent = markov_step(state, prev, context, w1, w)
    want_state, want_latent = ref_step(state, prev, context, w1, w)
    assert new_state.dtype == latent.dtype == torch.bfloat16 and new_state.shape == (rows, r)
    torch.testing.assert_close(new_state.double(), want_state, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(latent.double(), want_latent, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------- fold + softcap


def ref_fold(base, fw, fb, latent, cap):
    bias = bf16(fb.double() + latent.double() @ fw.double().T)
    x = bf16(base.double() + bias)
    return (cap * torch.tanh(x / cap)).bfloat16()


@pytest.mark.parametrize("rows,V,r", [(1, 300, 32), (3, 513, 64), (8, 200, 32)])
def test_fold_softcap_matches_addmm_then_add_softcap(rows, V, r):
    from radiance_inkling2_draft import fold_softcap

    g = torch.Generator().manual_seed(4)
    n_spec = 7
    # base_logits[:, i] of a (rows, n_spec, V) block: rows stride n_spec * V
    block = (torch.randn(rows, n_spec, V, generator=g) * 12).bfloat16()
    base = block[:, 2]
    fw = (torch.randn(V, r, generator=g) * 0.3).bfloat16().float()
    fb = torch.randn(V, generator=g).bfloat16().float()
    latent = torch.tanh(torch.randn(rows, r, generator=g)).bfloat16()
    out = fold_softcap(base, fw, fb, latent, 30.0)
    assert out.dtype == torch.bfloat16 and out.shape == (rows, V) and out.is_contiguous()
    torch.testing.assert_close(out.float(), ref_fold(base.float(), fw, fb, latent, 30.0).float(),
                               atol=0.25, rtol=2e-2)   # bf16 ulp of a logit near 30 is 0.125


# ---------------------------------------------------------------- shortlist (drafter-only change)


def make_fold(rows, V, r, seed=5):
    g = torch.Generator().manual_seed(seed)
    base = (torch.randn(rows, V, generator=g) * 6).bfloat16()
    fw = (torch.randn(V, r, generator=g) * 0.3).bfloat16().float()
    fb = torch.randn(V, generator=g).bfloat16().float()
    latent = torch.tanh(torch.randn(rows, r, generator=g)).bfloat16()
    return base, fw, fb, latent


@pytest.mark.parametrize("rows,V,r,k", [(1, 300, 32, 20), (3, 513, 64, 70), (2, 200, 32, 64)])
def test_gathered_fold_equals_dense_on_the_shortlist_and_minus_inf_elsewhere(rows, V, r, k):
    """The shortlist's entries get exactly the dense fold_softcap value, everything else is -inf: the
    draft distribution is the softmax over the shortlist, which the rejection sampler is handed as such
    (lossless for the target's output whatever the shortlist)."""
    from radiance_inkling2_draft import fold_softcap, fold_softcap_gather

    base, fw, fb, latent = make_fold(rows, V, r)
    vals, idx = base.float().topk(k, dim=-1)
    out = fold_softcap_gather(vals.bfloat16(), idx, fw, fb, latent, 30.0, V)
    dense = fold_softcap(base, fw, fb, latent, 30.0)
    member = torch.zeros(rows, V, dtype=torch.bool).scatter_(1, idx, True)
    assert out.dtype == torch.bfloat16 and out.shape == (rows, V) and out.is_contiguous()
    torch.testing.assert_close(out[member].float(), dense[member].float(), atol=0.25, rtol=2e-2)
    assert torch.isneginf(out[~member].float()).all()


def test_gathered_fold_never_reads_the_fold_weight_of_non_members():
    """The point of the shortlist: the dense pass streams the whole 103 MB fold weight per draft step.
    NaN planted in the rows of non-members must not leak anywhere."""
    from radiance_inkling2_draft import fold_softcap_gather

    rows, V, r, k = 2, 512, 32, 40
    base, fw, fb, latent = make_fold(rows, V, r)
    vals, idx = base.float().topk(k, dim=-1)
    member = torch.zeros(rows, V, dtype=torch.bool).scatter_(1, idx, True)
    fw[~member.any(0)] = float("nan")
    fb[~member.any(0)] = float("nan")
    out = fold_softcap_gather(vals.bfloat16(), idx, fw, fb, latent, 30.0, V)
    assert torch.isfinite(out[member].float()).all()


def test_gathered_fold_takes_a_row_strided_shortlist_view():
    """vals/idx of draft step i are slices [:, i] of the (rows, n_spec, k) topk result."""
    from radiance_inkling2_draft import fold_softcap, fold_softcap_gather

    rows, V, r, k, n_spec = 2, 300, 32, 30, 7
    base, fw, fb, latent = make_fold(rows, V, r)
    stack = torch.stack([base] * n_spec, 1)
    vals, idx = stack.float().topk(k, dim=-1)
    out = fold_softcap_gather(vals.bfloat16()[:, 3], idx[:, 3], fw, fb, latent, 30.0, V)
    dense = fold_softcap(base, fw, fb, latent, 30.0)
    member = torch.zeros(rows, V, dtype=torch.bool).scatter_(1, idx[:, 3], True)
    torch.testing.assert_close(out[member].float(), dense[member].float(), atol=0.25, rtol=2e-2)


@pytest.mark.parametrize("rows,n_spec,V,k", [(1, 7, 300, 20), (3, 7, 513, 64)])
def test_shortlist_is_the_topk_of_every_draft_row(rows, n_spec, V, k):
    from radiance_inkling2_draft import shortlist

    g = torch.Generator().manual_seed(9)
    base = (torch.randn(rows, n_spec, V, generator=g) * 6).bfloat16()
    vals, idx = shortlist(base, k)
    assert vals.shape == idx.shape == (rows, n_spec, k)
    torch.testing.assert_close(vals.float(), base.float().gather(-1, idx), rtol=0, atol=0)
    want = base.float().sort(-1, descending=True).values[..., :k]
    torch.testing.assert_close(vals.float(), want, rtol=0, atol=0)


@pytest.mark.parametrize("k", [0, 300, 400])
def test_shortlist_declines_when_it_cannot_shrink_anything(k):
    from radiance_inkling2_draft import shortlist

    assert shortlist(torch.zeros(1, 7, 300).bfloat16(), k) is None


# ---------------------------------------------------------------- int8 fold weight (RADIANCE_INKLING2_FOLD_INT8, drafter-only, lossy)


def _fold_inputs(rows, V, r, seed=6):
    g = torch.Generator().manual_seed(seed)
    base = (torch.randn(rows, V, generator=g) * 6).bfloat16()
    fw = (torch.randn(V, r, generator=g) * 0.3).bfloat16().float()
    fb = torch.randn(V, generator=g).bfloat16().float()
    latent = torch.tanh(torch.randn(rows, r, generator=g)).bfloat16()
    return base, fw, fb, latent


def _dq(fw):
    """Independent int8 round trip: per-row symmetric, scale = amax / 127, round to nearest."""
    s = fw.abs().amax(1, keepdim=True) / 127.0
    return (fw / s).round().clamp(-127, 127) * s


@pytest.fixture()
def int8_on(monkeypatch):
    import radiance_inkling2_draft as d
    monkeypatch.setattr(d, "FOLD_INT8", True)
    d._FOLD_Q8.clear()
    yield d
    d._FOLD_Q8.clear()


def test_fold_int8_knob_defaults_off():
    import radiance_inkling2_draft as d
    assert d.FOLD_INT8 is False


@pytest.mark.parametrize("rows,V,r", [(1, 300, 32), (3, 513, 64), (8, 200, 256)])
def test_fold_int8_equals_the_fold_of_the_dequantized_weight(int8_on, rows, V, r):
    base, fw, fb, latent = _fold_inputs(rows, V, r)
    out = int8_on.fold_softcap(base, fw, fb, latent, 30.0)
    want = ref_fold(base.float(), _dq(fw), fb, latent, 30.0).float()
    torch.testing.assert_close(out.float(), want, atol=0.25, rtol=2e-2)  # bf16 ulp of a logit near 30


@pytest.mark.parametrize("rows,V,r", [(1, 300, 32), (8, 400, 256)])
def test_fold_int8_error_vs_the_bf16_fold_is_bounded(int8_on, monkeypatch, rows, V, r):
    """Weight error <= scale / 2 per entry, so |d bias| <= sum_k |latent_k| * scale / 2; the softcap only
    shrinks it (|tanh'| <= 1). The bound is checked, not assumed."""
    base, fw, fb, latent = _fold_inputs(rows, V, r, seed=8)
    q = int8_on.fold_softcap(base, fw, fb, latent, 30.0).float()
    monkeypatch.setattr(int8_on, "FOLD_INT8", False)
    full = int8_on.fold_softcap(base, fw, fb, latent, 30.0).float()
    bound = (latent.float().abs().sum(1, keepdim=True) * (fw.abs().amax(1) / 127.0 / 2)[None, :])
    assert ((q - full).abs() <= bound + 0.25 + 0.0).all()  # + one bf16 ulp near the cap
    assert not torch.equal(q, full)  # the knob really changes the weights read


def test_fold_int8_reads_one_int8_copy_cached_per_weight(int8_on):
    base, fw, fb, latent = _fold_inputs(2, 300, 32)
    int8_on.fold_softcap(base, fw, fb, latent, 30.0)
    (key, (ref, q, s)), = int8_on._FOLD_Q8.items()
    assert q.dtype == torch.int8 and q.shape == fw.shape and s.dtype == torch.float32 and s.shape == (300,)
    assert (q.abs().amax(1) == 127).all() and torch.equal(s * 127, fw.abs().amax(1))  # amax / 127 per row
    int8_on.fold_softcap(base, fw, fb, latent, 30.0)
    assert len(int8_on._FOLD_Q8) == 1 and int8_on._FOLD_Q8[key][1] is q


def test_fold_int8_never_allocates_under_graph_capture(int8_on, monkeypatch):
    """A weight first seen during capture stays on the bf16 kernel (the copy is made by an eager warmup)."""
    monkeypatch.setattr(int8_on, "_capturing", lambda: True)
    base, fw, fb, latent = _fold_inputs(2, 300, 32)
    out = int8_on.fold_softcap(base, fw, fb, latent, 30.0)
    assert int8_on._FOLD_Q8 == {}
    torch.testing.assert_close(out.float(), ref_fold(base.float(), fw, fb, latent, 30.0).float(),
                               atol=0.25, rtol=2e-2)


def test_fold_int8_off_changes_nothing(monkeypatch):
    import radiance_inkling2_draft as d
    assert d.FOLD_INT8 is False
    base, fw, fb, latent = _fold_inputs(3, 513, 64)
    out = d.fold_softcap(base, fw, fb, latent, 30.0)
    torch.testing.assert_close(out.float(), ref_fold(base.float(), fw, fb, latent, 30.0).float(),
                               atol=0.25, rtol=2e-2)
    assert d._FOLD_Q8 == {}


def test_fold_int8_proof_line_once_per_weight(int8_on, capsys):
    int8_on._LOGGED.clear()
    base, fw, fb, latent = _fold_inputs(2, 300, 32)
    int8_on.fold_softcap(base, fw, fb, latent, 30.0)
    int8_on.fold_softcap(base, fw, fb, latent, 30.0)
    assert capsys.readouterr().out.count("[radiance.inkling2] FOLD_INT8 active") == 1


def test_fold_int8_shortlist_note(monkeypatch, capsys):
    """FOLD_INT8 + DRAFT_SHORTLIST: the gather path never reads the full fold weight, so the knob is inert; say so."""
    import radiance_inkling2_draft as d
    monkeypatch.setattr(d, "FOLD_INT8", True)
    monkeypatch.setattr(d, "SHORTLIST_K", 4096)
    d._LOGGED.clear()
    d.log_knobs()
    assert "FOLD_INT8=1 has no effect with DRAFT_SHORTLIST" in capsys.readouterr().out

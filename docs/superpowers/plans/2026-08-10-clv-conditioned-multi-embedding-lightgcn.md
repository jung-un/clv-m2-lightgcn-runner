# CLV-Conditioned Multi-Embedding LightGCN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validation-safe M2 runner in which CLV-related user behavior routes three user–item embedding experts while LightGCN graph propagation and plain BPR remain unchanged.

**Architecture:** Reuse the train-only future-value encoder and M1 data/evaluation pipeline. Add focused feature, model, and runner modules: a user/item profile builder, a dense CLV-gated mixture of embedding adapters, and a joint-warm orchestration layer with external-M1 controls and protected evaluation.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pandas, SciPy, scikit-learn, pytest, nbformat-compatible Colab JSON.

## Global Constraints

- Keep M2, M3, and M4 as equal thesis axes; this implementation changes representation only.
- Use the M1 binary graph and LightGCN propagation without CLV edge weights.
- Use plain, unweighted BPR with the existing negative sampler; no CLV sample weights or margins.
- Use `K=3`, expert hidden dimension 32, expert output dimension 16, gate hidden dimension 32.
- Use `lambda_train=1.0` and validation sweep `[0.0, 0.1, 0.25, 0.5, 1.0, 2.0]`.
- Primary training is M1 warm-start: five frozen epochs, then adapter/gate LR `5e-4` and LightGCN LR `5e-5`, at most 100 epochs with patience 20.
- Default to seed 42, validation only, `EVAL_TEST=False`, `EVAL_HOLDOUT=False`.
- Keep official validation/test/holdout out of encoder training, feature fitting, model selection, and hyperparameter selection.
- Select a positive lambda only if Recall and NDCG at 10/20/50 each remain within 1% relative of external M1; maximize price/purchase-amount weighted hit@10 and prefer the smaller lambda on ties.
- Save existing accuracy, economic, exposure, and segment metrics plus MoE gate/specialization diagnostics.
- Do not start a high-cost run until the Colab preflight and one-shot review gate are explicitly acknowledged.

---

## File Structure

- Create `clv_moe_features.py`: compose train-only user CLV profiles and item economic/category features.
- Create `clv_moe_model.py`: implement the three-expert embedding adapters, CLV gate, controls, scoring, and diagnostics.
- Create `lightgcn_clv_moe.py`: configure, train, evaluate, checkpoint, select lambda, and save results.
- Create `test_clv_moe_features.py`: data leakage and feature contract tests.
- Create `test_clv_moe_model.py`: embedding/gating math, controls, and M2/M4 boundary tests.
- Create `test_lightgcn_clv_moe.py`: training schedule, selection, persistence, and preflight tests.
- Create `clv_moe_colab.ipynb`: fresh-runtime clone/import, atomic presets, preflight, approval, run, and result display.
- Modify `RESEARCH_STATUS.md` after implementation verification to record code status without claiming experiment success.

---

### Task 1: Train-only CLV and item feature artifacts

**Files:**
- Create: `clv_moe_features.py`
- Create: `test_clv_moe_features.py`
- Reuse: `lightgcn_clv_residual.py`

**Interfaces:**
- Consumes: `residual.EncoderArtifact`, `residual.AnchorExamples`, `residual.transform_features()`, train DataFrame with `u_idx`, `i_idx`, `cat_idx`, `up`, `v`.
- Produces: `UserProfileArtifact`, `ItemProfileArtifact`, `compose_user_profiles(...)`, `build_item_profiles(...)`.

- [ ] **Step 1: Write failing user-profile tests**

```python
def test_user_profile_contains_behavior_masks_hidden_and_predictions():
    artifact, snapshot = _encoder_artifact_and_snapshot()
    out = features.compose_user_profiles(artifact, snapshot, torch.device("cpu"))
    assert out.values.shape == (snapshot.user_ids.max() + 2, 51)
    assert out.valid_user[snapshot.user_ids].all()
    assert np.isfinite(out.values).all()

def test_user_profile_does_not_mutate_when_validation_rows_change():
    train, validation = _train_and_validation_frames()
    before = _build_profiles_from_train(train)
    validation.loc[:, "v"] *= 1000
    after = _build_profiles_from_train(train)
    np.testing.assert_array_equal(before.values, after.values)
```

- [ ] **Step 2: Run user-profile tests and verify failure**

Run: `pytest -q test_clv_moe_features.py -k user_profile`

Expected: FAIL because `clv_moe_features` and `compose_user_profiles` do not exist.

- [ ] **Step 3: Implement user-profile artifact and composition**

```python
@dataclass(frozen=True)
class UserProfileArtifact:
    values: np.ndarray
    valid_user: np.ndarray
    feature_names: tuple[str, ...]

def compose_user_profiles(artifact, snapshot, device):
    transformed = residual.transform_features(snapshot, artifact.transform)
    x = torch.as_tensor(transformed, dtype=torch.float32, device=device)
    artifact.model.eval()
    with torch.no_grad():
        h, logit, log_amount = artifact.model(x)
        probability = torch.sigmoid(logit)
        log_ev = torch.log1p(probability * torch.expm1(log_amount))
    local = np.concatenate([
        transformed,
        h.cpu().numpy(),
        probability.cpu().numpy()[:, None],
        log_amount.cpu().numpy()[:, None],
        log_ev.cpu().numpy()[:, None],
    ], axis=1).astype(np.float32)
    n_users = artifact.h_all.shape[0]
    values = np.zeros((n_users, local.shape[1]), np.float32)
    valid = np.zeros(n_users, bool)
    values[snapshot.user_ids] = local
    valid[snapshot.user_ids] = True
    names = residual.NUMERIC_FEATURES + tuple(f"valid_{name}" for name in residual.NUMERIC_FEATURES)
    names += tuple(f"encoder_h_{j}" for j in range(16)) + (
        "future_purchase_probability", "future_log_amount", "future_log_ev"
    )
    return UserProfileArtifact(values, valid, names)
```

- [ ] **Step 4: Write failing item-profile tests**

```python
def test_item_profiles_are_train_only_finite_and_category_encoded():
    train = _tiny_train()
    out = features.build_item_profiles(train, n_items=5)
    assert out.numeric.shape == (5, 6)
    assert out.category_ids.shape == (5,)
    assert out.n_categories == int(out.category_ids.max()) + 1
    assert np.isfinite(out.numeric).all()

def test_item_profile_price_percentiles_ignore_validation():
    train = _tiny_train()
    before = features.build_item_profiles(train, 5)
    validation = train.copy()
    validation["up"] = 9999.0
    after = features.build_item_profiles(train, 5)
    np.testing.assert_array_equal(before.numeric, after.numeric)
```

- [ ] **Step 5: Implement item-profile artifact**

```python
@dataclass(frozen=True)
class ItemProfileArtifact:
    numeric: np.ndarray
    category_ids: np.ndarray
    valid_item: np.ndarray
    numeric_names: tuple[str, ...]
    n_categories: int

def build_item_profiles(train: pd.DataFrame, n_items: int) -> ItemProfileArtifact:
    item = train.groupby("i_idx", sort=False).agg(
        price=("up", "mean"), rows=("i_idx", "size"), users=("u_idx", "nunique"),
        category=("cat_idx", lambda x: x.mode().iat[0]),
    )
    pair = train.groupby(["u_idx", "i_idx"], sort=False).size()
    repeats = pair.gt(1).groupby(level="i_idx").mean().rename("repeat_share")
    item = item.join(repeats, how="left").fillna({"repeat_share": 0.0})
    item["price_percentile"] = item.price.rank(pct=True)
    item["category_price_percentile"] = item.groupby("category").price.rank(pct=True)
    item["user_percentile"] = item.users.rank(pct=True)
    raw = np.column_stack([
        item.price_percentile, item.category_price_percentile,
        np.log1p(item.rows), item.user_percentile, item.repeat_share,
        np.log1p(item.price),
    ]).astype(np.float32)
    mean, std = raw.mean(0), raw.std(0)
    std[std < 1e-6] = 1.0
    raw = (raw - mean) / std
    numeric = np.zeros((n_items, raw.shape[1]), np.float32)
    valid = np.zeros(n_items, bool)
    indices = item.index.to_numpy(dtype=np.int64)
    numeric[indices], valid[indices] = raw, True
    category_ids = np.zeros(n_items, np.int64)
    categories = pd.Index(sorted(pd.unique(item.category)))
    category_ids[indices] = categories.get_indexer(item.category) + 1
    names = ("price_percentile", "category_price_percentile", "log_rows",
             "user_percentile", "repeat_share", "log_price")
    return ItemProfileArtifact(numeric, category_ids, valid, names, len(categories) + 1)
```

- [ ] **Step 6: Run feature tests**

Run: `pytest -q test_clv_moe_features.py`

Expected: all feature tests PASS.

- [ ] **Step 7: Commit feature artifacts**

```bash
git add clv_moe_features.py test_clv_moe_features.py
git commit -m "feat: add train-only CLV MoE feature artifacts"
```

---

### Task 2: Mixture-of-embedding experts and controls

**Files:**
- Create: `clv_moe_model.py`
- Create: `test_clv_moe_model.py`

**Interfaces:**
- Consumes: a LightGCN-compatible base model exposing `embeddings()`, `UserProfileArtifact`, `ItemProfileArtifact`.
- Produces: `CLVMixtureEmbeddingModel`, `score_all()`, `bpr_loss()`, `moe_diagnostics()`.

- [ ] **Step 1: Write failing shape, gate, and lambda-zero tests**

```python
def test_three_experts_generate_user_and_item_embeddings():
    model = _model()
    ue, ie, gate = model.expert_embeddings()
    assert ue.shape == (4, 3, 16)
    assert ie.shape == (6, 3, 16)
    assert gate.shape == (4, 3)
    torch.testing.assert_close(gate.sum(1), torch.ones(4))

def test_lambda_zero_exactly_equals_external_m1_scores():
    model = _model()
    users = torch.arange(4)
    base = model.base_score_all(users)
    torch.testing.assert_close(model.score_all(users, lam=0.0), base, rtol=0, atol=0)
```

- [ ] **Step 2: Run model tests and verify failure**

Run: `pytest -q test_clv_moe_model.py -k 'three_experts or lambda_zero'`

Expected: FAIL because `CLVMixtureEmbeddingModel` does not exist.

- [ ] **Step 3: Implement embedding experts and gate**

```python
class EmbeddingExpert(nn.Module):
    def __init__(self, user_in: int, item_in: int):
        super().__init__()
        self.user = nn.Sequential(nn.Linear(user_in, 32), nn.GELU(), nn.Linear(32, 16))
        self.item = nn.Sequential(nn.Linear(item_in, 32), nn.GELU(), nn.Linear(32, 16))
        nn.init.normal_(self.user[-1].weight, std=0.01)
        nn.init.normal_(self.item[-1].weight, std=0.01)
        nn.init.zeros_(self.user[-1].bias)
        nn.init.zeros_(self.item[-1].bias)

class CLVMixtureEmbeddingModel(nn.Module):
    def __init__(self, base_model, user_profile, item_profile, control="clv"):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("user_profile", torch.as_tensor(user_profile.values, dtype=torch.float32))
        self.register_buffer("has_profile", torch.as_tensor(user_profile.valid_user, dtype=torch.float32))
        self.register_buffer("item_numeric", torch.as_tensor(item_profile.numeric, dtype=torch.float32))
        self.item_category = nn.Embedding(item_profile.n_categories, 8)
        self.register_buffer("item_category_ids", torch.as_tensor(item_profile.category_ids, dtype=torch.long))
        base_user, base_item, *_ = base_model.embeddings()
        self.experts = nn.ModuleList([
            EmbeddingExpert(base_user.shape[1] + user_profile.values.shape[1],
                            base_item.shape[1] + item_profile.numeric.shape[1] + 8)
            for _ in range(3)
        ])
        self.gate = nn.Sequential(nn.Linear(user_profile.values.shape[1], 32), nn.GELU(), nn.Linear(32, 3))
        self.control = control

    def base_parameters(self):
        return list(self.base_model.parameters())

    def adapter_parameters(self):
        base_ids = {id(p) for p in self.base_model.parameters()}
        return [p for p in self.parameters() if id(p) not in base_ids]
```

- [ ] **Step 4: Write failing scoring and control tests**

```python
def test_score_is_mixture_of_expert_embedding_inner_products():
    model = _model()
    users = torch.tensor([0, 2])
    ue, ie, gate = model.expert_embeddings(users)
    expected = model.base_score_all(users) + torch.einsum("uk,ukd,ikd->ui", gate, ue, ie)
    torch.testing.assert_close(model.score_all(users, 1.0), expected)

def test_constant_gate_removes_user_specific_routing_only():
    model = _model(control="constant_gate")
    gate = model.routing_weights(torch.arange(4))
    torch.testing.assert_close(gate, gate[:1].expand_as(gate))

def test_shuffled_clv_is_seed_deterministic_and_nonidentity():
    a = _model(control="shuffled_clv", seed=42).routed_profile
    b = _model(control="shuffled_clv", seed=42).routed_profile
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, _model(control="clv").routed_profile)

def test_single_adapter_parameter_count_is_within_five_percent():
    mixture = _model(control="clv")
    single = _model(control="single_adapter")
    ratio = _trainable_count(single) / _trainable_count(mixture)
    assert 0.95 <= ratio <= 1.05
```

- [ ] **Step 5: Implement controls, scoring, and plain BPR**

```python
def bpr_loss(self, users, positives, negatives, lam=1.0):
    pos = self.score_pairs(users, positives, lam)
    neg = self.score_pairs(users, negatives, lam)
    return -F.logsigmoid(pos - neg).mean()

def score_all(self, users, lam):
    base = self.base_score_all(users)
    if lam == 0:
        return base
    user_experts, item_experts, weights = self.expert_embeddings(users)
    residual = torch.einsum("uk,ukd,ikd->ui", weights, user_experts, item_experts)
    return base + lam * self.has_profile[users, None] * residual
```

- [ ] **Step 6: Write and pass M2 boundary tests**

```python
def test_bpr_is_plain_mean_without_clv_weights():
    model = _model()
    u, i, j = torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([3, 4])
    expected = -F.logsigmoid(model.score_pairs(u, i, 1.0) - model.score_pairs(u, j, 1.0)).mean()
    torch.testing.assert_close(model.bpr_loss(u, i, j), expected)

def test_model_reuses_one_binary_base_graph_for_every_expert():
    base = _base_model()
    model = _model(base_model=base)
    assert all(not hasattr(expert, "adj") for expert in model.experts)
    assert model.base_model.adj is base.adj
```

Run: `pytest -q test_clv_moe_model.py`

Expected: all model tests PASS.

For `single_adapter`, search integer hidden widths from 1 through 512 and choose the width whose one-expert trainable
parameter count is closest to the three-expert model including its gate. Store the selected width and exact parameter
counts in diagnostics; fail construction if the ratio falls outside `[0.95, 1.05]`.

- [ ] **Step 7: Commit the model**

```bash
git add clv_moe_model.py test_clv_moe_model.py
git commit -m "feat: add CLV-gated mixture of embedding experts"
```

---

### Task 3: Joint-warm training, frozen diagnostic, and compute control

**Files:**
- Create: `lightgcn_clv_moe.py`
- Create: `test_lightgcn_clv_moe.py`
- Reuse: `lightgcn_clv_v3.py`, `lightgcn_clv_residual.py`

**Interfaces:**
- Produces: `MoEConfig`, `configure_moe_run()`, `train_moe()`, `train_pref_continue()`, `state_hash()`.

- [ ] **Step 1: Write failing configuration and schedule tests**

```python
def test_default_screening_is_seed42_validation_only():
    cfg = moe.configure_moe_run("dunnhumby")
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is False and cfg.eval_holdout is False
    assert cfg.expert_count == 3 and cfg.frozen_epochs == 5
    assert cfg.adapter_lr == 5e-4 and cfg.base_lr == 5e-5
    assert cfg.lambda_eval == (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)

def test_joint_warm_updates_only_adapters_before_epoch_six():
    model, data, cfg = _tiny_training_case(max_epochs=6)
    records = moe.train_moe(model, data, _base_cfg(), cfg, seed=42, eval_recall=_constant_recall)
    assert records["base_updates_by_epoch"][:5] == [0, 0, 0, 0, 0]
    assert records["base_updates_by_epoch"][5] > 0
```

- [ ] **Step 2: Run schedule tests and verify failure**

Run: `pytest -q test_lightgcn_clv_moe.py -k 'default_screening or joint_warm'`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement configuration and parameter groups**

```python
@dataclass
class MoEConfig:
    dataset: str = "dunnhumby"
    seed_list: tuple[int, ...] = (42,)
    expert_count: int = 3
    frozen_epochs: int = 5
    max_epochs: int = 100
    patience: int = 20
    adapter_lr: float = 5e-4
    base_lr: float = 5e-5
    lambda_train: float = 1.0
    lambda_eval: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    accuracy_tolerance: float = 0.01
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str | None = None
    m1_checkpoint_dir: str | None = None

def optimizer_for(model, cfg):
    return torch.optim.Adam([
        {"params": model.adapter_parameters(), "lr": cfg.adapter_lr},
        {"params": model.base_parameters(), "lr": cfg.base_lr},
    ])

def set_base_trainable(model, enabled):
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(enabled)

def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

def plain_bpr_batches(data, base_cfg, rng):
    order = rng.permutation(len(data["tr_u"]))
    device = data["adj"].device
    for start in range(0, len(order), int(base_cfg["BATCH_SIZE"])):
        idx = order[start:start + int(base_cfg["BATCH_SIZE"])]
        bu, bi = data["tr_u"][idx], data["tr_i"][idx]
        bj = v3.sample_negatives(bu, bi, data["n_items"], data["pos_key"], rng,
                                 base_cfg["NEG_MODE"], data["item_cat"], data["cat_items"])
        yield tuple(torch.as_tensor(x, dtype=torch.long, device=device) for x in (bu, bi, bj))
```

- [ ] **Step 4: Implement deterministic joint-warm training**

```python
def train_moe(model, data, base_cfg, cfg, seed, eval_recall, freeze_base=False):
    residual._seed_everything(seed)
    rng = np.random.default_rng(seed)
    best_state, best, bad = None, -float("inf"), 0
    records = {"base_updates_by_epoch": [], "updates": 0, "samples": 0, "loss": "plain_bpr"}
    optimizer = optimizer_for(model, cfg)
    for epoch in range(1, cfg.max_epochs + 1):
        base_active = (not freeze_base) and epoch > cfg.frozen_epochs
        set_base_trainable(model, base_active)
        base_updates = 0
        for bu, bi, bj in plain_bpr_batches(data, base_cfg, rng):
            loss = model.bpr_loss(bu, bi, bj, cfg.lambda_train)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            base_updates += int(base_active)
            records["updates"] += 1; records["samples"] += len(bu)
        records["base_updates_by_epoch"].append(base_updates)
        score = float(eval_recall(model))
        if score > best + 1e-12:
            best_state, best, bad = clone_state(model), score, 0
        else:
            bad += 1
        if bad >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return records | {"best_val_recall@10": best}
```

- [ ] **Step 5: Write and implement frozen and pref-continue controls**

```python
def test_frozen_moe_preserves_m1_hash():
    model, data, cfg = _tiny_training_case()
    before = moe.state_hash(model.base_model)
    moe.train_moe(model, data, _base_cfg(), cfg, seed=42, eval_recall=_constant_recall, freeze_base=True)
    assert moe.state_hash(model.base_model) == before

def test_pref_continue_has_matched_base_updates_and_no_adapter():
    base, data, cfg = _tiny_pref_case()
    stats = moe.train_pref_continue(base, data, cfg, seed=42, target_base_updates=7)
    assert stats["base_updates"] == 7
    assert stats["loss"] == "plain_bpr"
```

Run: `pytest -q test_lightgcn_clv_moe.py -k 'frozen_moe or pref_continue'`

Expected: PASS after implementing `freeze_base` and `train_pref_continue` with the same sampler and LR.

- [ ] **Step 6: Commit the training layer**

```bash
git add lightgcn_clv_moe.py test_lightgcn_clv_moe.py
git commit -m "feat: add joint-warm CLV MoE training controls"
```

---

### Task 4: Protected evaluation, model selection, and MoE diagnostics

**Files:**
- Modify: `clv_moe_model.py`
- Modify: `lightgcn_clv_moe.py`
- Modify: `test_clv_moe_model.py`
- Modify: `test_lightgcn_clv_moe.py`

**Interfaces:**
- Produces: `select_lambda()`, `moe_diagnostics()`, `validate_result_metrics()`, `run_experiment()`.

- [ ] **Step 1: Write failing six-guardrail selection tests**

```python
def test_select_lambda_uses_all_recall_ndcg_guardrails():
    base = _baseline_metrics()
    rows = [_row(0.0, base), _row(0.5, base, revenue10=1.1),
            _row(1.0, base, recall50=base["recall@50"] * 0.989, revenue10=1.2)]
    selected, table = moe.select_lambda(rows, base, tolerance=0.01)
    assert selected == 0.5
    assert not bool(table.loc[table["lambda"].eq(1.0), "eligible"].iat[0])

def test_lambda_zero_fallback_is_not_success():
    selected, table = moe.select_lambda([_row(0.0, _baseline_metrics())], _baseline_metrics())
    assert selected == 0.0
    assert table.attrs["success"] is False
```

- [ ] **Step 2: Implement selection by adapting the validated residual selector**

```python
def select_lambda(rows, baseline, tolerance=0.01):
    table = pd.DataFrame(rows).copy()
    checks = []
    for k in (10, 20, 50):
        for metric in ("recall", "ndcg"):
            col = f"{metric}@{k}"
            checks.append(table[col] >= baseline[col] * (1.0 - tolerance))
    table["eligible"] = np.logical_and.reduce(checks)
    candidates = table[table.eligible & table["lambda"].gt(0)]
    selected = 0.0 if candidates.empty else float(
        candidates.sort_values(["revenue@10", "lambda"], ascending=[False, True]).iloc[0]["lambda"]
    )
    table.attrs["success"] = selected > 0
    return selected, table
```

- [ ] **Step 3: Write failing specialization diagnostics tests**

```python
def test_moe_diagnostics_reports_gate_usage_entropy_and_expert_similarity():
    diag = model_module.moe_diagnostics(_model(), torch.arange(4), sample_items=5)
    assert set(diag) >= {"gate_entropy_mean", "expert_usage_mean", "expert_embedding_cosine",
                         "expert_score_correlation", "residual_to_base_score_std"}
    assert len(diag["expert_usage_mean"]) == 3
    assert np.isfinite(diag["gate_entropy_mean"])
```

- [ ] **Step 4: Implement deterministic bounded diagnostics**

```python
@torch.no_grad()
def moe_diagnostics(model, users, sample_items=2048):
    users = users[: min(len(users), 2048)]
    ue, ie, gate = model.expert_embeddings(users)
    entropy = -(gate * torch.log(gate.clamp_min(1e-12))).sum(1)
    item = ie[: min(len(ie), sample_items)]
    expert_scores = torch.einsum("ukd,ikd->kui", ue, item)
    return {
        "gate_entropy_mean": float(entropy.mean()),
        "expert_usage_mean": gate.mean(0).cpu().tolist(),
        "expert_embedding_cosine": pairwise_expert_cosines(ue, item),
        "expert_score_correlation": pairwise_score_correlations(expert_scores),
        "residual_to_base_score_std": residual_base_std_ratio(model, users, item.shape[0]),
    }
```

- [ ] **Step 5: Implement protected orchestration and persistence**

`run_experiment()` must perform this exact sequence:

```python
cfg = cfg or configure_moe_run("dunnhumby")
base_cfg = pure_m1_config(cfg)  # pref_only, binary, plain, uniform
data = v3.prepare_data(base_cfg, v3.DCFG)
anchors, snapshot = train_only_encoder_inputs(data, cfg)
for seed in cfg.seed_list:
    encoder = residual.train_future_value_encoder(anchors, snapshot, encoder_cfg(cfg), seed, v3.DEVICE)
    user_profile = compose_user_profiles(encoder, snapshot, v3.DEVICE)
    item_profile = build_item_profiles(data["train"], data["n_items"])
    base_model = load_external_m1(seed, data, base_cfg)
    main_model = build_moe(base_model, user_profile, item_profile, control="clv")
    train_moe(main_model, data, base_cfg, cfg, seed, validation_recall_callback)
    evaluate_and_store_all_validation_lambdas(main_model, external_m1=base_model)
selected = select_each_model_against_external_m1()
if selected["clv_moe"]["success"]:
    for control in ("pref_continue", "frozen_moe", "constant_gate", "shuffled_clv", "single_adapter"):
        train_evaluate_and_store_seed42_control(control)
if cfg.eval_test:
    evaluate_only_validation_selected_lambdas_on_test()
save_csv_json_checkpoints_and_delta_tables()
```

The JSON must include `cfg`, `base_cfg`, input feature names, selected lambda and success flag, encoder diagnostics,
training records, MoE diagnostics, checkpoint paths, absolute rows, paired deltas, and result fingerprint.

- [ ] **Step 6: Test protected splits and result schema**

```python
def test_eval_test_false_never_constructs_test_answers(monkeypatch):
    seen = []
    monkeypatch.setattr(v3, "prepare_data", _prepare_data_spy(seen))
    moe.run_experiment(moe.configure_moe_run("dunnhumby", eval_test=False, max_epochs=1))
    assert "test" not in seen

def test_result_rows_include_exposure_and_moe_diagnostics():
    required = {"n_distinct@10", "coverage@10", "exposure_entropy@10", "eff_catalog@10",
                "top10_share@10", "top100_share@10", "gate_entropy_mean"}
    assert required <= set(_one_saved_row())
```

Run: `pytest -q test_lightgcn_clv_moe.py test_clv_moe_model.py`

Expected: all tests PASS.

- [ ] **Step 7: Commit evaluation and orchestration**

```bash
git add clv_moe_model.py lightgcn_clv_moe.py test_clv_moe_model.py test_lightgcn_clv_moe.py
git commit -m "feat: add protected CLV MoE evaluation pipeline"
```

---

### Task 5: Colab runner and high-cost review gate

**Files:**
- Create: `clv_moe_colab.ipynb`
- Modify: `test_lightgcn_clv_moe.py`

**Interfaces:**
- Consumes: `configure_moe_run()`, `preflight_summary()`, `run_experiment()`.

- [ ] **Step 1: Write failing notebook structure test**

```python
def test_colab_has_clone_path_preflight_and_high_cost_gate():
    notebook = json.loads(Path("clv_moe_colab.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "feat/clv-residual-lightgcn" in source
    assert "sys.path.insert" in source
    assert "configure_moe_run" in source
    assert "ACKNOWLEDGE_HIGH_COST = False" in source
    assert "assert ACKNOWLEDGE_HIGH_COST" in source
    assert "eval_test=False" in source and "eval_holdout=False" in source
```

- [ ] **Step 2: Run notebook test and verify failure**

Run: `pytest -q test_lightgcn_clv_moe.py -k colab`

Expected: FAIL because `clv_moe_colab.ipynb` does not exist.

- [ ] **Step 3: Create the notebook with these cells**

1. Mount Drive and clone/pull branch `feat/clv-residual-lightgcn`.
2. Insert the repository path into `sys.path` and import `lightgcn_clv_moe`.
3. Select `DATASET = "dunnhumby"` or `"hm"` and atomically configure seed 42, validation-only paths.
4. Assert CUDA and display `preflight_summary(cfg)` without training.
5. Set `ACKNOWLEDGE_HIGH_COST = False`, assert it, then call `run_experiment(cfg)`.
6. Display selected lambdas, success flags, absolute metrics, paired deltas, exposure metrics, and saved paths.

- [ ] **Step 4: Validate notebook JSON and tests**

Run: `python -m json.tool clv_moe_colab.ipynb >/dev/null && pytest -q test_lightgcn_clv_moe.py -k colab`

Expected: JSON validation succeeds and notebook test PASS.

- [ ] **Step 5: Commit Colab runner**

```bash
git add clv_moe_colab.ipynb test_lightgcn_clv_moe.py
git commit -m "feat: add guarded CLV MoE Colab runner"
```

---

### Task 6: Full verification, documentation, and handoff

**Files:**
- Modify: `RESEARCH_STATUS.md` in the workspace root.
- Modify only if required by verification: files created in Tasks 1–5.

**Interfaces:**
- Produces: verified branch and a one-shot high-cost review summary; does not run the real high-cost experiment.

- [ ] **Step 1: Run focused tests**

Run: `pytest -q test_clv_moe_features.py test_clv_moe_model.py test_lightgcn_clv_moe.py`

Expected: all new tests PASS.

- [ ] **Step 2: Run the full suite and lint**

Run: `pytest -q && ruff check lightgcn_clv_moe.py clv_moe_features.py clv_moe_model.py test_clv_moe_features.py test_clv_moe_model.py test_lightgcn_clv_moe.py`

Expected: the full suite passes and ruff reports no errors.

- [ ] **Step 3: Run CPU smoke tests on both dataset presets**

Run:

```bash
python - <<'PY'
import lightgcn_clv_moe as m
for dataset in ("dunnhumby", "hm"):
    cfg = m.configure_moe_run(dataset, max_epochs=1, encoder_epochs=1)
    summary = m.preflight_summary(cfg)
    assert summary["seed_list"] == [42]
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    assert summary["expert_count"] == 3
print("preset smoke passed")
PY
```

Expected: `preset smoke passed`.

- [ ] **Step 4: Update research status with implementation facts only**

Record the commit, created runner/notebook, test counts, smoke-test outcome, and that no real H&M/Dunnhumby result exists yet. Keep the design hypothesis under “잠정” and do not claim improvement.

- [ ] **Step 5: Verify the final diff and repository state**

Run: `git diff --check && git status --short && git log --oneline -8`

Expected: no whitespace errors; only the intentional workspace-root `RESEARCH_STATUS.md` change may remain outside this repository.

- [ ] **Step 6: Commit final documentation fixes**

```bash
git add lightgcn_clv_moe.py clv_moe_features.py clv_moe_model.py clv_moe_colab.ipynb \
  test_clv_moe_features.py test_clv_moe_model.py test_lightgcn_clv_moe.py
git commit -m "docs: finalize CLV MoE experiment handoff"
```

If the index is empty because all code was committed task-by-task, do not create an empty commit.

- [ ] **Step 7: Present the one-shot high-cost review**

Report model equation and controls, exact dataset/seed/split settings, input/output paths, expected checkpoints and CSV/JSON files, tests run, and the unchanged `ACKNOWLEDGE_HIGH_COST=False`. Stop before real training so the user can review all changes at once.

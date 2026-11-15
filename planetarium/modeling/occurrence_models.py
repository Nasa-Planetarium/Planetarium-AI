"""Model adapters for binary next-month hotspot verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from planetarium.config import SEED
from planetarium.modeling.forecast_benchmark import (
    BASELINES,
    CAL_SAMPLE,
    DATES,
    FIT_SAMPLE,
    LOOKBACK,
    NEURAL,
    sample,
)

TIMESFM = ("google/timesfm-3.0-pytorch", "43046b85ec22d584a13f8098c2ed39c889e129c2")
TIMESFM_LOOKBACK = 24


def _timesfm_scores(evaluator, histories):
    import torch

    # Same decoding and postprocessing as the public evaluator for finite,
    # nonnegative univariate contexts and one forecast step.
    target = torch.as_tensor(histories[:, None], device=evaluator.device)
    with torch.inference_mode():
        logits = evaluator.model.decode(target=target, horizon=1)[:, 0, 0, :]
        ordered = logits.sort(dim=-1).values
        return (
            ordered[:, evaluator.config.median_quantile_index]
            .clamp_min(0)
            .cpu()
            .numpy()
        )


def timesfm_signal(panel, folder: Path, device: str, threads: int) -> np.ndarray:
    import torch
    from huggingface_hub import snapshot_download
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    path = folder / f"timesfm3_signal_{TIMESFM_LOOKBACK}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    torch.set_num_threads(threads)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    first = int(np.searchsorted(DATES, 2019 * 12))
    states = np.empty((len(DATES) - first, panel.width), dtype=np.uint64)
    history = np.zeros(panel.width, dtype=np.uint64)
    mask = np.uint64((1 << TIMESFM_LOOKBACK) - 1)
    for target in range(1, len(DATES)):
        history = ((history << np.uint64(1)) & mask) | (panel.counts[target - 1] > 0)
        if target >= first:
            states[target - first] = history
    keys, aligned = np.unique(states, return_inverse=True)
    checkpoint = snapshot_download(
        TIMESFM[0],
        revision=TIMESFM[1],
        allow_patterns=["config.json", "model.safetensors"],
    )
    batch = 2048
    evaluator = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=checkpoint, per_core_batch_size=batch, device=device
        )
    )
    point = np.empty(len(keys), dtype=np.float32)
    shifts = np.arange(TIMESFM_LOOKBACK - 1, -1, -1, dtype=np.int64).astype(np.uint64)
    print(f"TimesFM 3.0: {len(keys):,} unique binary histories", flush=True)
    for i in range(0, len(keys), batch):
        histories = ((keys[i : i + batch, None] >> shifts) & np.uint64(1)).astype(
            np.float32
        )
        point[i : i + batch] = _timesfm_scores(evaluator, histories)
        if i % (batch * 25) == 0:
            print(
                f"TimesFM 3.0: {min(i + batch, len(keys)):,}/{len(keys):,}", flush=True
            )
    if not np.isfinite(point).all():
        raise ValueError("TimesFM returned a non-finite score.")
    score = np.full((len(DATES), panel.width), np.nan, dtype=np.float32)
    score[first:] = point[aligned].reshape(states.shape)
    with path.with_suffix(".tmp").open("wb") as handle:
        np.save(handle, score)
    path.with_suffix(".tmp").replace(path)
    return np.load(path, mmap_mode="r")


def _tabular(name: str, threads: int, device: str):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=100))

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=140,
            learning_rate=0.10,
            num_leaves=31,
            max_bin=63,
            n_jobs=threads,
            verbosity=-1,
            random_state=SEED,
            force_col_wise=True,
        )

    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=140,
            learning_rate=0.10,
            max_depth=6,
            max_bin=63,
            device=device,
            tree_method="hist",
            objective="binary:logistic",
            n_jobs=threads,
            random_state=SEED,
        )

    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=140,
            learning_rate=0.10,
            depth=6,
            border_count=63,
            random_seed=SEED,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model {name}")


def _neural(name, panel, train_months, threads, device):
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    channels = 2
    width = 20 if name == "patchtsmixer" else 24
    if name == "patchtsmixer":
        from transformers import PatchTSMixerConfig, PatchTSMixerModel

        backbone = PatchTSMixerModel(
            PatchTSMixerConfig(
                context_length=LOOKBACK,
                num_input_channels=channels,
                patch_length=12,
                patch_stride=6,
                d_model=width,
                num_layers=2,
                expansion_factor=2,
                scaling=None,
            )
        )
    else:
        from transformers import PatchTSTConfig, PatchTSTModel

        backbone = PatchTSTModel(
            PatchTSTConfig(
                context_length=LOOKBACK,
                num_input_channels=channels,
                patch_length=12,
                patch_stride=6,
                d_model=width,
                num_hidden_layers=2,
                num_attention_heads=4,
                ffn_dim=48,
                scaling=None,
            )
        )

    # Concatenate the learned past-window representation with the same causal
    # seasonal and geographic inputs used by the tabular classifiers.
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            with torch.no_grad():
                size = backbone(
                    past_values=torch.zeros(1, LOOKBACK, channels)
                ).last_hidden_state.numel()
            self.head = nn.Sequential(
                nn.Linear(size + panel.x.shape[-1], 48), nn.GELU(), nn.Linear(48, 1)
            )

        def forward(self, history, x):
            encoded = self.backbone(past_values=history).last_hidden_state.flatten(1)
            return self.head(torch.cat((encoded, x), dim=-1)).squeeze(-1)

    counts = torch.as_tensor(panel.counts.astype(np.float32), device=device)
    logged = torch.log1p(counts)
    offsets = torch.arange(-LOOKBACK, 0, device=device)
    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)

    def inputs(ids):
        target, grid = np.divmod(ids, panel.width)
        tt = torch.as_tensor(target, device=device)
        gg = torch.as_tensor(grid, device=device)
        history = logged[tt[:, None] + offsets, gg[:, None]]
        h = torch.stack((history / 4, (history > 0).float()), dim=-1)
        x = torch.as_tensor(panel.x[target - LOOKBACK, grid], device=device)
        return h, x

    for _ in range(240):
        t = rng.choice(train_months, size=1024)
        g = rng.integers(panel.width, size=1024)
        y = torch.as_tensor((panel.counts[t, g] > 0).astype(np.float32), device=device)
        loss = F.binary_cross_entropy_with_logits(
            model(*inputs(t * panel.width + g)), y
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()

    def raw(ids):
        with torch.inference_mode():
            return np.concatenate(
                [
                    torch.sigmoid(model(*inputs(ids[i : i + 4096]))).cpu().numpy()
                    for i in range(0, len(ids), 4096)
                ]
            )

    return raw


def make_predictor(
    name, panel, train_months, calibration_months, threads, device, timesfm
):
    if name in BASELINES:
        index = 0 if name == "seasonal" else 1
        return lambda t: panel.x[t - LOOKBACK, :, index]

    if name == "timesfm3":

        def raw(ids):
            t, g = np.divmod(ids, panel.width)
            return timesfm[t, g]

        def transform(values):
            return values.reshape(-1, 1)

    else:
        if name in NEURAL:
            raw = _neural(name, panel, train_months, threads, device)
        else:
            fit = sample(train_months, panel.width, FIT_SAMPLE, SEED)
            x, y = panel.lookup(fit)
            estimator = _tabular(name, threads, device).fit(x, y)

            def raw(ids):
                t, g = np.divmod(ids, panel.width)
                xx = panel.x[t - LOOKBACK, g]
                if name == "lightgbm":
                    return estimator.booster_.predict(xx, num_threads=threads)
                return estimator.predict_proba(xx)[:, 1]

        def transform(values):
            return logit(values.astype(np.float64).clip(1e-6, 1 - 1e-6)).reshape(-1, 1)

    cal = sample(calibration_months, panel.width, CAL_SAMPLE, SEED + 1)
    tc, gc = np.divmod(cal, panel.width)
    y_cal = panel.counts[tc, gc] > 0
    calibrator = LogisticRegression(max_iter=100).fit(transform(raw(cal)), y_cal)

    def predict(target):
        ids = target * panel.width + np.arange(panel.width)
        return calibrator.predict_proba(transform(raw(ids)))[:, 1]

    return predict

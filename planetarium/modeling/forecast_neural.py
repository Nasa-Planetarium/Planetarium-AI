"""Pooled direct neural models for monthly hotspot forecasting."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from planetarium.config import SEED
from planetarium.modeling.forecast_candidates import training_folds

CONTEXT = 36
EPOCHS = 4
LEARNING_RATE = 0.003


class DLinear(nn.Module):
    def __init__(self, horizon, kernel_size=13):
        super().__init__()
        self.padding = kernel_size // 2
        self.smoother = nn.AvgPool1d(kernel_size, stride=1)
        self.seasonal = nn.Linear(CONTEXT, horizon)
        self.trend = nn.Linear(CONTEXT, horizon)
        for layer in (self.seasonal, self.trend):
            nn.init.constant_(layer.weight, 1 / CONTEXT)
            nn.init.zeros_(layer.bias)

    def forward(self, context):
        mean = self.smoother(
            F.pad(context[:, None], (self.padding, self.padding), mode="replicate")
        )[:, 0]
        return self.seasonal(context - mean) + self.trend(mean)


def _nhits(horizon):
    from neuralforecast.models import NHITS

    # Nixtla's published architecture, trained below with the same dated
    # windows and optimizer used for DLinear.
    return NHITS(
        h=horizon,
        input_size=CONTEXT,
        n_blocks=[1, 1, 1],
        mlp_units=[[128, 128]] * 3,
        n_pool_kernel_size=[6, 3, 1],
        n_freq_downsample=[3, 2, 1],
        random_seed=SEED,
    )


def _estimator(model, horizon):
    if model == "dlinear":
        return DLinear(horizon), {"kernel_size": 13}

    settings = {"package": f"neuralforecast=={version('neuralforecast')}"}
    if model == "nhits":
        return _nhits(horizon), settings
    if model == "tide":
        from neuralforecast.models import TiDE

        architecture = {
            "hidden_size": 128,
            "decoder_output_dim": 16,
            "temporal_decoder_dim": 32,
            "dropout": 0.1,
        }
        return TiDE(
            h=horizon, input_size=CONTEXT, random_seed=SEED, **architecture
        ), settings | architecture

    from neuralforecast.models import PatchTST

    architecture = {
        "patch_len": 12,
        "stride": 6,
        "encoder_layers": 2,
        "n_heads": 4,
        "hidden_size": 64,
        "linear_hidden_size": 128,
        "dropout": 0.1,
        "fc_dropout": 0.1,
    }
    return PatchTST(
        h=horizon, input_size=CONTEXT, random_seed=SEED, **architecture
    ), settings | architecture


def _forward(estimator, model, context):
    if model == "dlinear":
        return estimator(context)
    return estimator(
        {
            "insample_y": context[..., None],
            "insample_mask": torch.ones_like(context[..., None]),
            "futr_exog": None,
            "hist_exog": None,
            "stat_exog": None,
        }
    )[..., 0]


def train(model, panel, horizon, device, threads, batch_size):
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.set_float32_matmul_precision("high")

    values = torch.tensor(np.log1p(panel.values), device=device)
    forecast = np.empty((len(panel.origins), len(panel.ids), horizon), dtype=np.float32)
    for cutoff, rows in training_folds(panel):
        assert cutoff <= panel.origins[rows].min()
        windows = values[:, : cutoff + 1].unfold(1, CONTEXT + horizon, 1)
        n_windows = windows.shape[1]
        n_examples = len(values) * n_windows
        estimator, architecture = _estimator(model, horizon)
        estimator = estimator.to(device)
        optimizer = torch.optim.AdamW(estimator.parameters(), lr=LEARNING_RATE)
        rng = torch.Generator(device=device).manual_seed(SEED)
        for _ in range(EPOCHS):
            order = torch.randperm(n_examples, device=device, generator=rng)
            for batch in order.split(batch_size):
                sample = windows[batch // n_windows, batch % n_windows]
                fitted = _forward(estimator, model, sample[:, :CONTEXT])
                loss = F.l1_loss(fitted, sample[:, CONTEXT:])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        print(
            f"{model} fit through {panel.dates[cutoff]} ({n_examples:,} windows)",
            flush=True,
        )
        estimator.eval()
        with torch.inference_mode():
            for row in rows:
                origin = panel.origins[row]
                inputs = values[:, origin + 1 - CONTEXT : origin + 1]
                prediction = torch.cat(
                    [
                        _forward(estimator, model, batch)
                        for batch in inputs.split(batch_size)
                    ]
                )
                forecast[row] = torch.expm1(prediction).cpu().numpy()
    return forecast, {
        "fit": "annual",
        "target": "log1p",
        "loss": "MAE",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "context": CONTEXT,
        "seed": SEED,
        "torch": str(torch.__version__),
        **architecture,
    }

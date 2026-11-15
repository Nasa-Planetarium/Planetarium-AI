"""Adapters for frozen foundation checkpoints used in the monthly benchmark."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np

CHECKPOINTS = {
    "timesfm3": (
        "google/timesfm-3.0-pytorch",
        "43046b85ec22d584a13f8098c2ed39c889e129c2",
    ),
    "chronos2": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"),
    "timestral": (
        "mistralai/Timestral-209M",
        "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
    ),
    "tirex2": ("NX-AI/TiRex-2", "352636173653592b6dceb3fbeee9224ca97952a9"),
    "timesfm25": (
        "google/timesfm-2.5-200m-pytorch",
        "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    ),
    "chronos_bolt": (
        "amazon/chronos-bolt-base",
        "5d9f166d69f47aef3401367a7b842e78fe97b121",
    ),
    "toto2": ("Datadog/Toto-2.0-22m", "685e4ae3e2be8d8998025e53dd98e7fdcb296a89"),
    "moirai2": (
        "Salesforce/moirai-2.0-R-small",
        "30f43ff08c8494f4943ae1521e9d4e94a0fbb389",
    ),
}


def _timesfm3(checkpoint, device, batch_size, horizon):
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    evaluator = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=checkpoint,
            per_core_batch_size=batch_size,
            device=device,
        )
    )

    def infer(context):
        outputs = list(
            evaluator.predict_batch(
                list(context),
                horizon=horizon,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )
        quantiles = np.asarray(
            [result.quantiles for result in outputs], dtype=np.float32
        )
        return quantiles[..., [0, 4, 8]]

    return infer, {
        "package": f"timesfm=={version('timesfm')}",
        "symmetric_averaging": False,
    }


def _chronos2(checkpoint, device, batch_size, horizon):
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        checkpoint, device_map=device, dtype=torch.float32
    )

    def infer(context):
        quantiles, _ = pipeline.predict_quantiles(
            np.ascontiguousarray(context[:, None, :]),
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=batch_size,
            cross_learning=False,
        )
        return np.stack([result[0].cpu().numpy() for result in quantiles])

    return infer, {
        "package": f"chronos-forecasting=={version('chronos-forecasting')}",
        "cross_learning": False,
    }


def _timestral(checkpoint, device, _batch_size, horizon):
    from timestral import TimestralForecaster

    forecaster = TimestralForecaster.from_pretrained(checkpoint, device=device)

    def infer(context):
        forecast = forecaster.forecast(
            context[:, None, :], prediction_length=horizon, symmetric_averaging=False
        )
        indices = [forecast.quantile_levels.index(q) for q in (0.1, 0.5, 0.9)]
        return forecast.quantiles[:, 0][..., indices].cpu().numpy()

    return infer, {
        "package": f"timestral=={version('timestral')}",
        "code_revision": "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff",
        "symmetric_averaging": False,
    }


def _tirex2(checkpoint, device, batch_size, horizon):
    import torch
    from tirex2 import TimeseriesType, load_model

    model = load_model(checkpoint, device=device, cuda_graphs=False)

    def infer(context):
        inputs = [
            TimeseriesType(row, None, None)
            for row in torch.from_numpy(context[:, None, :].copy())
        ]
        forecasts = model.forecast(
            inputs,
            prediction_length=horizon,
            batch_size=batch_size,
            output_type="numpy",
            tta_sign_flip=False,
        )
        return np.stack([forecast[0, [0, 4, 8], :].T for forecast in forecasts])

    return infer, {"package": f"tirex-2=={version('tirex-2')}", "tta_sign_flip": False}


def _timesfm25(checkpoint, device, batch_size, horizon):
    import torch
    from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

    model = TimesFM_2p5_200M_torch(torch_compile=False)
    model.model.device = torch.device(device)
    model.model.device_count = 1
    model.load_checkpoint(checkpoint)
    model.compile(
        ForecastConfig(
            max_context=192,
            max_horizon=horizon,
            normalize_inputs=True,
            per_core_batch_size=batch_size,
            use_continuous_quantile_head=True,
        )
    )

    def infer(context):
        _, quantiles = model.forecast(horizon=horizon, inputs=list(context))
        return quantiles[..., [1, 5, 9]]

    return infer, {"package": f"timesfm=={version('timesfm')}", "max_context": 192}


def _chronos_bolt(checkpoint, device, _batch_size, horizon):
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        checkpoint, device_map=device, dtype=torch.float32
    )

    def infer(context):
        quantiles, _ = pipeline.predict_quantiles(
            torch.from_numpy(np.ascontiguousarray(context)),
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        return quantiles.cpu().numpy()

    return infer, {"package": f"chronos-forecasting=={version('chronos-forecasting')}"}


def _toto2(checkpoint, device, _batch_size, horizon):
    import torch
    from toto2 import Toto2Model

    model = Toto2Model.from_pretrained(checkpoint).to(device).eval()
    patch_size = model.config.patch_size
    indices = [model.output_head.knots.index(q) for q in (0.1, 0.5, 0.9)]

    def infer(context):
        target = torch.tensor(context[:, None, :], device=device)
        padding = (-target.shape[-1]) % patch_size
        mask = torch.ones_like(target, dtype=torch.bool)
        target = torch.nn.functional.pad(target, (padding, 0))
        mask = torch.nn.functional.pad(mask, (padding, 0), value=False)
        quantiles = model.forecast(
            {
                "target": target,
                "target_mask": mask,
                "series_ids": torch.zeros(
                    (len(target), 1), device=device, dtype=torch.long
                ),
            },
            horizon=horizon,
            decode_block_size=None,
            has_missing_values=bool(padding),
        )
        return quantiles[indices, :, 0].permute(1, 2, 0).cpu().numpy()

    return infer, {
        "package": f"toto-2=={version('toto-2')}",
        "decode_block_size": None,
        "context_padding": "left masked",
        "patch_size": patch_size,
    }


def _moirai2(checkpoint, device, _batch_size, horizon):
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    model = (
        Moirai2Forecast(
            module=Moirai2Module.from_pretrained(checkpoint),
            prediction_length=horizon,
            context_length=192,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        .to(device)
        .eval()
    )

    def infer(context):
        target = torch.tensor(context[..., None], device=device)
        with model.hparams_context(context_length=context.shape[1]):
            quantiles = model(
                past_target=target,
                past_observed_target=torch.ones_like(target, dtype=torch.bool),
                past_is_pad=torch.zeros(
                    target.shape[:2], device=device, dtype=torch.bool
                ),
            )
        return quantiles[:, [0, 4, 8]].permute(0, 2, 1).cpu().numpy()

    return infer, {"package": f"uni2ts=={version('uni2ts')}"}


ADAPTERS = {
    "timesfm3": _timesfm3,
    "chronos2": _chronos2,
    "timestral": _timestral,
    "tirex2": _tirex2,
    "timesfm25": _timesfm25,
    "chronos_bolt": _chronos_bolt,
    "toto2": _toto2,
    "moirai2": _moirai2,
}


def predict(model, panel, horizon, device, threads, batch_size):
    import torch
    from huggingface_hub import snapshot_download

    torch.set_num_threads(threads)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.set_float32_matmul_precision("high")
    repo, revision = CHECKPOINTS[model]
    checkpoint = snapshot_download(
        repo, revision=revision, allow_patterns=["config.json", "model.safetensors"]
    )
    infer, config = ADAPTERS[model](checkpoint, device, batch_size, horizon)
    quantiles = np.empty(
        (len(panel.origins), len(panel.ids), horizon, 3), dtype=np.float32
    )
    with torch.inference_mode():
        for row, origin in enumerate(panel.origins):
            for first in range(0, len(panel.ids), batch_size):
                last = first + batch_size
                quantiles[row, first:last] = infer(
                    panel.values[first:last, : origin + 1]
                )
            if row == 0 or (row + 1) % 6 == 0 or row == len(panel.origins) - 1:
                print(
                    f"{model} {row + 1}/{len(panel.origins)} {panel.dates[origin]}",
                    flush=True,
                )
    quantiles.sort(axis=-1)
    return (
        quantiles[..., 1].copy(),
        quantiles[..., 0].copy(),
        quantiles[..., 2].copy(),
        {
            **config,
            "checkpoint": repo,
            "revision": revision,
            "torch": str(torch.__version__),
        },
    )

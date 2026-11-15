"""Pinned Hugging Face adapters for daily value histories."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np

CHECKPOINTS = {
    "timesfm3": (
        "google/timesfm-3.0-pytorch",
        "43046b85ec22d584a13f8098c2ed39c889e129c2",
    ),
    "chronos2": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"),
    "chronos_bolt": (
        "amazon/chronos-bolt-base",
        "5d9f166d69f47aef3401367a7b842e78fe97b121",
    ),
    "tirex2": ("NX-AI/TiRex-2", "352636173653592b6dceb3fbeee9224ca97952a9"),
}


def _timesfm3(checkpoint, device, batch_size, horizon):
    import torch
    from timesfm3 import TimesFM3Forecaster

    forecaster = TimesFM3Forecaster.from_pretrained(
        checkpoint, device=device, per_core_batch_size=batch_size
    )
    patch = forecaster.config.output_patch_length
    padded_horizon = ((horizon + patch - 1) // patch) * patch
    mixed_precision = device == "cuda" and torch.cuda.is_bf16_supported()

    def infer(context):
        target = torch.from_numpy(context[:, None, :]).to(device)
        # These fixed-width histories are already finite, univariate and patch-aligned.
        with torch.autocast(
            device_type=device, enabled=mixed_precision, dtype=torch.bfloat16
        ):
            quantiles = forecaster.model.decode(target, horizon=padded_horizon)
        return (
            quantiles[:, 0, :horizon]
            .float()
            .sort(dim=-1)
            .values[..., [0, 4, 8]]
            .cpu()
            .numpy()
        )

    return infer, f"timesfm=={version('timesfm')}"


def _chronos(model, checkpoint, device, batch_size, horizon):
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        checkpoint, device_map=device, dtype=torch.float32
    )

    def infer(context):
        options = {"prediction_length": horizon, "quantile_levels": [0.1, 0.5, 0.9]}
        if model == "chronos2":
            quantiles, _ = pipeline.predict_quantiles(
                context[:, None, :],
                batch_size=batch_size,
                cross_learning=False,
                **options,
            )
            return torch.stack([result[0] for result in quantiles]).cpu().numpy()
        quantiles, _ = pipeline.predict_quantiles(torch.from_numpy(context), **options)
        return quantiles.cpu().numpy()

    return infer, f"chronos-forecasting=={version('chronos-forecasting')}"


def _tirex2(checkpoint, device, batch_size, horizon):
    import torch
    from tirex2 import TimeseriesType, load_model

    model = load_model(checkpoint, device=device, cuda_graphs=False)

    def infer(context):
        inputs = [
            TimeseriesType(row, None, None)
            for row in torch.from_numpy(context[:, None, :])
        ]
        forecasts = model.forecast(
            inputs,
            prediction_length=horizon,
            batch_size=batch_size,
            output_type="numpy",
            tta_sign_flip=False,
        )
        return np.stack([forecast[0, [0, 4, 8], :].T for forecast in forecasts])

    return infer, f"tirex-2=={version('tirex-2')}"


def load_inference(model, horizon, device, batch_size, threads):
    import torch
    from huggingface_hub import snapshot_download

    torch.set_num_threads(threads)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.set_float32_matmul_precision("high")
    repo, revision = CHECKPOINTS[model]
    checkpoint = snapshot_download(
        repo, revision=revision, allow_patterns=["config.json", "model.safetensors"]
    )
    options = (checkpoint, device, batch_size, horizon)
    if model == "timesfm3":
        infer, package = _timesfm3(*options)
    elif model == "tirex2":
        infer, package = _tirex2(*options)
    else:
        infer, package = _chronos(model, *options)

    def predict(context):
        unique, inverse = np.unique(
            np.ascontiguousarray(context, dtype=np.float32), axis=0, return_inverse=True
        )
        forecasts = np.empty((len(unique), horizon, 3), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(unique), batch_size):
                end = min(start + batch_size, len(unique))
                forecast = infer(unique[start:end])
                if (
                    forecast.shape != (end - start, horizon, 3)
                    or not np.isfinite(forecast).all()
                ):
                    raise ValueError(
                        f"{model} returned invalid quantiles: {forecast.shape}"
                    )
                forecasts[start:end] = forecast
        return forecasts[inverse]

    return predict, {
        "package": package,
        "torch": str(torch.__version__),
        "revision": revision,
    }

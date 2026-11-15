"""TimesFM 3.0 다음 달 분위수 예측 (별도 실험 환경에서 실행: torch + timesfm@918df02, 가중치 google/timesfm-3.0-pytorch@43046b8).
라이선스: TimesFM Non-Commercial License v1.0 — 비교 실험(연구)에만 사용. 출력은 운영·앱에 쓰지 않는다.

입력  <dir>/series_{task}.npz (V[t, i] = log1p(월 value), t = 2012-02 ~ 2025-12)
출력  <dir>/tfm_{task}_{mode}.npz : q[k, i, h, j] = 기준 달 k(마지막 관측 달), 계열 i, h+1개월 뒤, 분위수 j (log1p 척도), levels, origins
      mode: ctx36 (직전 36개월) | full (2012-02부터 직전 달까지)
실행  python timesfm_forecast.py <dir> <checkpoint_dir> grid 2020 2025
설계 참고: 팀원 Jonghun Shin(ozingmw) firms-1km-backend의 TimesFM 호출(같은 입력은 한 번만 계산, bf16 autocast).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

D, CKPT, TASK, Y0, Y1 = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
BATCH = 512
ck = next(iter(CKPT.glob("*.safetensors")))
ev = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(ck), per_core_batch_size=BATCH, device="cuda"))
model = ev.model
levels = np.array(ev.config.quantiles, dtype=np.float32)
hp = ev.config.output_patch_length
bf16 = torch.cuda.is_bf16_supported()
print("quantiles", levels, "output patch", hp, "bf16", bf16, flush=True)

z = np.load(D / f"series_{TASK}.npz")
V, T = z["V"], z["t"]
H = 3                                   # 1·2·3개월 뒤
targets = [t for t in T if Y0 <= t // 12 <= Y1]
# 기준 시점(마지막 관측 달) o: 대상 달 = o + h (h = 1..3) 이 시험 기간에 들도록
origins = [o for o in T if any(Y0 <= (o + h) // 12 <= Y1 for h in range(1, H + 1)) and o + 1 <= T[-1]]


def decode(ctx: np.ndarray):
    ctx = np.ascontiguousarray(ctx, dtype=np.float32)
    keys = ctx.view(np.dtype((np.void, ctx.shape[1] * 4))).ravel()
    _, rows, inv = np.unique(keys, return_index=True, return_inverse=True)
    out = np.empty((len(rows), H, len(levels)), np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        for s in range(0, len(rows), BATCH):
            b = rows[s:s + BATCH]
            x = torch.from_numpy(ctx[b, None, :]).to("cuda")
            r = model.decode(x, horizon=hp)
            out[s:s + len(b)] = r[:, 0, :H].float().cpu().numpy()
    return np.sort(out, axis=-1)[inv], len(rows)


for mode in ("ctx36", "full"):
    t0 = time.perf_counter()
    Q = np.empty((len(origins), V.shape[1], H, len(levels)), np.float16)
    uniq = 0
    for k, o in enumerate(origins):
        i = int(o - T[0]) + 1                 # V[:i] = 2012-02 ~ o
        ctx = V[max(0, i - 36):i].T if mode == "ctx36" else V[:i].T   # 36개월이 안 되는 초기 기준 달은 있는 만큼
        Q[k], u = decode(ctx)
        uniq += u
        if k % 12 == 11:
            print(f"[{TASK} {mode}] {k + 1}/{len(origins)} origins, unique contexts {uniq:,}, {time.perf_counter() - t0:,.0f}s, "
                  f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    np.savez_compressed(D / f"tfm_{TASK}_{mode}.npz", q=Q, levels=levels, origins=np.array(origins))
    print(f"[done] {TASK} {mode}: {len(origins)} origins x {V.shape[1]:,} series x {H} horizons, unique {uniq:,}, "
          f"{time.perf_counter() - t0:,.0f}s", flush=True)

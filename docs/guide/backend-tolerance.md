# Backend Tolerance & Debugging

accel-gpu is designed to produce numerically consistent results across backend modes:

- WebGPU
- WebGL2
- CPU (JS / optional WASM runner)

## What "consistent" means

- Core ops are expected to match within floating-point tolerance, not bit-exact equality.
- Typical tolerances used in tests are `1e-4` to `2e-4` for real-valued outputs.
- Reduction-heavy or iterative operations can drift slightly more than simple elementwise math.

## Fallback behavior

Initialization order by default:

1. WebGPU
2. WebGL2
3. CPU

Override in tests or production diagnostics:

```ts
const cpu = await init({ forceCPU: true });
const webgl = await init({ forceWebGL: true });
const auto = await init();
```

## Recommended validation strategy

When validating model/data pipelines:

- Compare outputs from `forceCPU: true` and default `init()`.
- Use absolute tolerance assertions (`abs(a - b) <= eps`).
- Start with deterministic, fixed input tensors.

## Debug checklist

- Verify tensor shapes at operation boundaries.
- Re-run with `forceCPU: true` to isolate GPU/backend effects.
- Use smaller matrices/signals to localize first divergence.
- Check for power-of-two constraints in FFT/spectrogram APIs.

## Performance tuning notes

- Use WebGPU where available for large matrix and convolution workloads.
- Keep arrays alive only as long as needed; wrap temporary tensors in `gpu.tidy(...)`.
- Prefer subpath imports for smaller bundles:

```ts
import { matmul } from "accel-gpu/linalg";
import { fft } from "accel-gpu/signal";
```

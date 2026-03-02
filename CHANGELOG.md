# Changelog

All notable changes to @accel/gpu will be documented in this file.

## [0.2.0] - Unreleased

### Added

- **Shape inference** — `matmul(gpu, A, B)` infers M, N, K from array shapes
- **Method chaining** — `a.add(b).mul(2).sum()` returns `this` for chaining
- **reshape()** — Reshape arrays with shape metadata
- **CPU fallback** — Automatic fallback when WebGPU unavailable (Node, headless)
- **Buffer pooling** — Reuse GPUBuffers for better performance
- **fromImageData() / toCanvas()** — Canvas integration for image processing
- **layerNorm** — Layer normalization kernel for transformers
- **attentionScores** — Q @ K^T / sqrt(dim) for attention
- **Clear error messages** — Descriptive errors with shape info
- **Benchmark page** — Compare GPU vs CPU performance
- **Playground** — Interactive code editor
- **forceCPU** init option — Force CPU backend for testing

### Changed

- `init()` now uses WebGPU with automatic CPU fallback
- `matmul`, `softmax`, `transpose` support shape inference from array metadata

## [0.1.0] - Initial Release

- Core API: `init()`, `gpu.array()`, `toArray()`
- Math ops: `add`, `mul`, `sum`, `max`
- Linear algebra: `matmul`, `dot`, `transpose`
- ML: `softmax`
- WebGPU backend

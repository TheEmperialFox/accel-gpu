# API Reference

## Context

- `init(options?)`
- `gpu.array(data, shape?)`
- `gpu.zeros(shape)` / `gpu.ones(shape)` / `gpu.full(shape, value)`
- `gpu.arange(start, stop, step?)` / `gpu.linspace(start, stop, num)`
- `gpu.random(shape)` / `gpu.randn(shape)`
- `gpu.fromArrow(column, options?)`
- `gpu.fromBuffer(buffer, options?)`
- `gpu.scoped(fn)` / `gpu.tidy(fn)`

## Math Ops

- `add`, `sub`, `mul`, `div`, `sum`, `max`, `min`, `mean`

## Linear Algebra

- `matmul`, `dot`, `transpose`, `inv`, `det`, `solve`, `qr`, `svd`

## ML & Convolution

- `softmax`, `layerNorm`, `batchNorm`, `attentionScores`
- `conv2d`, `maxPool2d`, `avgPool2d`

## Signal Processing

- `fft`, `ifft`, `fftMagnitude`, `spectrogram`

## Training

- `gradients`, `sgdStep`

## Shape Notes

- `matmul(gpu, A, B)` infers dimensions from array shape metadata.
- `layerNorm(gpu, input, gamma, beta, rows?, cols?)` supports explicit row/column shape override.

## TypeScript Shape Safety

Use tuple shapes to get stronger compile-time checks in IDEs:

```ts
const A = gpu.array<[2, 3]>(new Float32Array(6), [2, 3]);
const B = gpu.array<[3, 4]>(new Float32Array(12), [3, 4]);
const C = await matmul(gpu, A, B); // GPUArray<[2, 4]>

const v1 = gpu.array<[3]>(new Float32Array([1, 2, 3]), [3]);
const v2 = gpu.array<[3]>(new Float32Array([4, 5, 6]), [3]);
const d = await dot(gpu, v1, v2);
```

## Live API Playground

<iframe src="https://phantasm0009.github.io/accel-gpu/playground/" width="100%" height="560" style="border:1px solid #ddd;border-radius:8px;"></iframe>

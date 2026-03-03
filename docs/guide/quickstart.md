# Quick Start

## Install

```bash
npm install accel-gpu
```

## Initialize

```ts
import { init } from "accel-gpu";

const gpu = await init();
console.log(gpu.backendType); // webgpu | webgl | cpu
```

## Basic Ops

```ts
const a = gpu.array([1, 2, 3, 4]);
const b = gpu.array([5, 6, 7, 8]);
await a.add(b).mul(2);
console.log(await a.sum());
```

## Memory Safety

```ts
await gpu.tidy(async (ctx) => {
  const tmp = ctx.array([1, 2, 3]);
  await tmp.mul(2);
});
```

## Zero-Copy Style Data Ingestion

```ts
import { fromArrow, fromBuffer } from "accel-gpu/data";

const a = fromArrow(gpu, arrowColumn, { shape: [rows, cols] });
const b = fromBuffer(gpu, sharedArrayBuffer, { shape: [rows, cols] });
```

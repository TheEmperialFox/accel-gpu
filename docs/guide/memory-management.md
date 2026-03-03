# Memory Management

## Why This Matters

WebGPU buffers live in VRAM. JavaScript garbage collection is not deterministic for GPU resources, so unbounded array creation can exhaust VRAM quickly.

## Option 1: Manual Disposal

```ts
const a = gpu.array([1, 2, 3]);
await a.mul(2);
a.dispose();
```

## Option 2: `gpu.tidy(...)` (Recommended)

`tidy` tracks arrays created inside the callback and disposes all intermediates automatically. Returned arrays survive.

```ts
const result = await gpu.tidy(async (ctx) => {
  const a = ctx.array([1, 2, 3]);
  const b = ctx.array([4, 5, 6]);
  await a.add(b).mul(2);
  return a;
});

console.log(await result.toArray());
result.dispose();
```

## Nested `tidy`

Nested tidy scopes are supported. Arrays returned by inner tidy calls stay alive for outer scopes and are still eligible for disposal when outer scopes finish.

## Option 3: Finalization Fallback

`GPUArray` uses `FinalizationRegistry` as a best-effort fallback to release leaked buffers when objects are garbage-collected.

Use `tidy` or explicit `dispose()` for predictable, production-safe memory behavior.

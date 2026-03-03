import { describe, it, expect } from "vitest";
import { init } from "../../dist/index.js";

describe("memory management", () => {
  it("tidy disposes intermediates and keeps returned GPUArray", async () => {
    const gpu = await init({ forceCPU: true });

    let intermediate: import("../../dist/index.js").GPUArray | undefined;
    const result = await gpu.tidy(async (ctx) => {
      const a = ctx.array([1, 2, 3]);
      const b = ctx.array([4, 5, 6]);
      intermediate = b;
      await a.add(b);
      return a;
    });

    expect(result.isDisposed).toBe(false);
    expect(intermediate?.isDisposed).toBe(true);
    expect(Array.from(await result.toArray())).toEqual([5, 7, 9]);
  });

  it("tidy supports nested scopes without disposing inner returned arrays too early", async () => {
    const gpu = await init({ forceCPU: true });

    let outerIntermediate: import("../../dist/index.js").GPUArray | undefined;

    const out = await gpu.tidy(async (ctx) => {
      const a = await ctx.tidy(async (inner) => {
        const x = inner.array([1, 2, 3]);
        const y = inner.array([1, 1, 1]);
        await x.add(y);
        return x;
      });

      expect(a.isDisposed).toBe(false);
      const b = ctx.array([2, 2, 2]);
      outerIntermediate = b;
      await a.add(b);
      return a;
    });

    expect(out.isDisposed).toBe(false);
    expect(outerIntermediate?.isDisposed).toBe(true);
    expect(Array.from(await out.toArray())).toEqual([4, 5, 6]);
  });
});

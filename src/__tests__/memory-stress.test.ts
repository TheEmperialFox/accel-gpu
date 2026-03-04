import { describe, expect, it } from "vitest";
import { init } from "../../dist/index.js";

describe("memory stress", () => {
  it("repeated tidy scopes dispose intermediates", async () => {
    const gpu = await init({ forceCPU: true });
    const survivors: import("../../dist/index.js").GPUArray[] = [];

    for (let i = 0; i < 80; i++) {
      const out = await gpu.tidy(async (ctx) => {
        const a = ctx.array([i, i + 1, i + 2]);
        const b = ctx.array([1, 2, 3]);
        await a.add(b);
        const c = await a.clone();
        return c;
      });
      survivors.push(out);
      expect(out.isDisposed).toBe(false);
    }

    for (const arr of survivors) {
      arr.dispose();
      expect(arr.isDisposed).toBe(true);
    }
  });

  it("nested scoped and tidy loops keep results valid", async () => {
    const gpu = await init({ forceCPU: true });

    for (let i = 0; i < 40; i++) {
      const value = await gpu.scoped(async (ctx) => {
        return ctx.tidy(async (inner) => {
          const a = inner.array([1, 2, 3, 4]);
          const b = inner.array([2, 2, 2, 2]);
          await a.mul(b);
          return a.sum();
        });
      });

      expect(value).toBe(20);
    }
  });
});

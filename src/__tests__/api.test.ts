import { describe, it, expect, beforeAll } from "vitest";
import { init, matmul, softmax, transpose, layerNorm } from "../../dist/index.js";

describe("accel-gpu API", () => {
  let gpu: Awaited<ReturnType<typeof init>>;

  beforeAll(async () => {
    gpu = await init({ forceCPU: true });
  });

  it("add", async () => {
    const a = gpu.array(new Float32Array([1, 2, 3, 4]));
    const b = gpu.array(new Float32Array([5, 6, 7, 8]));
    await a.add(b);
    const result = await a.toArray();
    expect(result[0]).toBe(6);
    expect(result[3]).toBe(12);
  });

  it("mul", async () => {
    const m = gpu.array(new Float32Array([2, 4, 6, 8]));
    await m.mul(2);
    const result = await m.toArray();
    expect(result[0]).toBe(4);
    expect(result[3]).toBe(16);
  });

  it("sum", async () => {
    const sum = await gpu.array(new Float32Array([1, 2, 3, 4])).sum();
    expect(sum).toBe(10);
  });

  it("max", async () => {
    const maxVal = await gpu.array(new Float32Array([1, 5, 3, 2])).max();
    expect(maxVal).toBe(5);
  });

  it("dot", async () => {
    const v1 = gpu.array(new Float32Array([1, 2, 3]));
    const v2 = gpu.array(new Float32Array([4, 5, 6]));
    const dotVal = await v1.dot(v2);
    expect(dotVal).toBe(32);
  });

  it("matmul", async () => {
    const A = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const B = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [3, 2]);
    const C = await matmul(gpu, A, B);
    const result = await C.toArray();
    const expected = [22, 28, 49, 64];
    expected.forEach((exp, i) => expect(result[i]).toBeCloseTo(exp, 5));
  });

  it("transpose", async () => {
    const T = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const Tt = await transpose(gpu, T, 2, 3);
    const result = await Tt.toArray();
    const expected = [1, 4, 2, 5, 3, 6];
    expected.forEach((exp, i) => expect(result[i]).toBeCloseTo(exp, 5));
  });

  it("softmax", async () => {
    const logits = gpu.array(new Float32Array([1, 2, 3, 4]));
    const probs = await softmax(gpu, logits);
    const result = await probs.toArray();
    const sum = result.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it("layerNorm", async () => {
    const lnInput = gpu.array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const gamma = gpu.array(new Float32Array([1, 1, 1]));
    const beta = gpu.array(new Float32Array([0, 0, 0]));
    const lnOut = await layerNorm(gpu, lnInput, gamma, beta, 2, 3);
    const result = await lnOut.toArray();
    expect(result.length).toBe(6);
    const row0Sum = result.slice(0, 3).reduce((a, b) => a + b, 0);
    const row1Sum = result.slice(3, 6).reduce((a, b) => a + b, 0);
    expect(row0Sum).toBeCloseTo(0, 5);
    expect(row1Sum).toBeCloseTo(0, 5);
  });
});

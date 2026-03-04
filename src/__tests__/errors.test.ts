import { describe, expect, it } from "vitest";
import { conv2d, fft, ifft, init, inv, matmul, spectrogram, transpose } from "../../dist/index.js";

describe("error message standardization", () => {
  it("matmul shape mismatch is descriptive", async () => {
    const gpu = await init({ forceCPU: true });
    const a = gpu.array([1, 2, 3, 4], [2, 2]);
    const b = gpu.array([1, 2, 3], [3, 1]);
    await expect(matmul(gpu, a, b)).rejects.toThrow(
      "matmul: A is 2×2, B is 3×1 — inner dimensions must match (2 vs 3)."
    );
  });

  it("transpose explicit mismatch uses standard length message", async () => {
    const gpu = await init({ forceCPU: true });
    const a = gpu.array([1, 2, 3]);
    await expect(transpose(gpu, a, 2, 2)).rejects.toThrow(
      "transpose: length mismatch — 4 vs 3. Arrays must have the same length."
    );
  });

  it("inverse requires square matrix", async () => {
    const gpu = await init({ forceCPU: true });
    const a = gpu.array([1, 2, 3, 4, 5, 6], [2, 3]);
    await expect(inv(gpu, a)).rejects.toThrow("inv: matrix must be square, got 2×3.");
  });

  it("fft length must be power of two", async () => {
    const gpu = await init({ forceCPU: true });
    const a = gpu.array([1, 2, 3]);
    await expect(fft(gpu, a)).rejects.toThrow("fft: length must be a power of 2, got 3.");
  });

  it("ifft requires interleaved even length", async () => {
    const gpu = await init({ forceCPU: true });
    const a = gpu.array([1, 0, 2]);
    await expect(ifft(gpu, a)).rejects.toThrow(
      "ifft: length 3 is invalid — expected even length for interleaved complex data (2*N)"
    );
  });

  it("spectrogram frameLength must be power of two", async () => {
    const gpu = await init({ forceCPU: true });
    const a = gpu.array([1, 2, 3, 4, 5, 6]);
    await expect(spectrogram(gpu, a, 3, 1)).rejects.toThrow(
      "spectrogram: frameLength must be a power of 2, got 3."
    );
  });

  it("conv2d validates kernel rank", async () => {
    const gpu = await init({ forceCPU: true });
    const input = gpu.array([1, 2, 3, 4], [2, 2, 1]);
    const kernel = gpu.array([1, 0, 0, 1], [2, 2]);
    await expect(conv2d(gpu, input, kernel)).rejects.toThrow(
      "conv2d kernel: expected rank-4 [kH,kW,C_in,C_out], got rank 2."
    );
  });
});

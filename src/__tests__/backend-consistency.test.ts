import { beforeAll, describe, expect, it } from "vitest";
import { conv2d, fft, init, inv, matmul } from "../../dist/index.js";

async function arrayClose(a: Float32Array, b: Float32Array, epsilon = 1e-4) {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(epsilon);
  }
}

describe("backend consistency", () => {
  let cpu: Awaited<ReturnType<typeof init>>;
  let auto: Awaited<ReturnType<typeof init>>;
  let wasmPreferred: Awaited<ReturnType<typeof init>>;

  beforeAll(async () => {
    cpu = await init({ forceCPU: true });
    auto = await init();
    wasmPreferred = await init({ forceCPU: true, preferWasmCPU: true });
  });

  it("matmul matches across backend selections", async () => {
    const dataA = new Float32Array([1, 2, 3, 4, 5, 6]);
    const dataB = new Float32Array([7, 8, 9, 10, 11, 12]);

    const cCpu = await matmul(cpu, cpu.array(dataA, [2, 3]), cpu.array(dataB, [3, 2]));
    const cAuto = await matmul(auto, auto.array(dataA, [2, 3]), auto.array(dataB, [3, 2]));
    const cWasm = await matmul(
      wasmPreferred,
      wasmPreferred.array(dataA, [2, 3]),
      wasmPreferred.array(dataB, [3, 2])
    );

    await arrayClose(await cCpu.toArray(), await cAuto.toArray());
    await arrayClose(await cCpu.toArray(), await cWasm.toArray());
  });

  it("inv matches across backend selections", async () => {
    const matrix = new Float32Array([4, 1, 2, 1, 3, 0, 2, 0, 5]);
    const iCpu = await inv(cpu, cpu.array(matrix, [3, 3]));
    const iAuto = await inv(auto, auto.array(matrix, [3, 3]));
    const iWasm = await inv(wasmPreferred, wasmPreferred.array(matrix, [3, 3]));

    await arrayClose(await iCpu.toArray(), await iAuto.toArray(), 2e-4);
    await arrayClose(await iCpu.toArray(), await iWasm.toArray(), 2e-4);
  });

  it("fft matches across backend selections", async () => {
    const signalData = new Float32Array([0, 1, 0, -1, 0, 1, 0, -1]);
    const fCpu = await fft(cpu, cpu.array(signalData));
    const fAuto = await fft(auto, auto.array(signalData));
    const fWasm = await fft(wasmPreferred, wasmPreferred.array(signalData));

    await arrayClose(await fCpu.toArray(), await fAuto.toArray(), 2e-4);
    await arrayClose(await fCpu.toArray(), await fWasm.toArray(), 2e-4);
  });

  it("conv2d matches across backend selections", async () => {
    const inputData = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
    ]);
    const kernelData = new Float32Array([
      1, 0,
      0, -1,
    ]);

    const outCpu = await conv2d(cpu, cpu.array(inputData, [4, 4, 1]), cpu.array(kernelData, [2, 2, 1, 1]));
    const outAuto = await conv2d(auto, auto.array(inputData, [4, 4, 1]), auto.array(kernelData, [2, 2, 1, 1]));
    const outWasm = await conv2d(
      wasmPreferred,
      wasmPreferred.array(inputData, [4, 4, 1]),
      wasmPreferred.array(kernelData, [2, 2, 1, 1])
    );

    await arrayClose(await outCpu.toArray(), await outAuto.toArray(), 2e-4);
    await arrayClose(await outCpu.toArray(), await outWasm.toArray(), 2e-4);
  });

  it("default init is deterministic with forced CPU in Node", () => {
    expect(auto.backendType).toBe("cpu");
  });
});

/**
 * Fallback logic: WebGPU -> WebGL -> CPU
 */

import { createWebGPUBackend } from "./webgpu";
import { createCPUBackend } from "./cpu-backend";
import { KernelRunner } from "./kernel-runner";
import { CPURunner } from "./cpu-runner";
import type { WebGPUBackend } from "./webgpu";
import type { CPUBackend } from "./cpu-backend";

export type Backend = WebGPUBackend | CPUBackend;
export type Runner = KernelRunner | CPURunner;

export async function createBackend(): Promise<{
  backend: Backend;
  runner: Runner;
  backendType: "webgpu" | "webgl" | "cpu";
}> {
  try {
    if (typeof navigator !== "undefined" && navigator.gpu) {
      const backend = await createWebGPUBackend();
      const runner = new KernelRunner(backend);
      return { backend, runner, backendType: "webgpu" };
    }
  } catch {
    // WebGPU failed, fall through to CPU
  }

  const backend = createCPUBackend();
  const runner = new CPURunner();
  return { backend, runner, backendType: "cpu" };
}

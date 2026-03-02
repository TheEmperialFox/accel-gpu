/**
 * Core types for Accel GPU
 */

import type { WebGPUBackend } from "./backend/webgpu";
import type { CPUBackend } from "./backend/cpu-backend";
import type { KernelRunner } from "./backend/kernel-runner";
import type { CPURunner } from "./backend/cpu-runner";
import type { GPUArray } from "./array";

export interface AccelContext {
  backend: WebGPUBackend | CPUBackend;
  runner: KernelRunner | CPURunner;
  /** Backend in use: 'webgpu' | 'webgl' | 'cpu' */
  backendType: "webgpu" | "webgl" | "cpu";
  array(data: Float32Array | number[], shape?: number[]): GPUArray;
  fromImageData(imageData: ImageData): GPUArray;
  toCanvas(arr: GPUArray, width: number, height: number): Promise<HTMLCanvasElement>;
}

/**
 * Core types for Accel GPU
 */

import type { WebGPUBackend } from "./backend/webgpu";
import type { WebGLBackend } from "./backend/webgl-backend";
import type { CPUBackend } from "./backend/cpu-backend";
import type { KernelRunner } from "./backend/kernel-runner";
import type { WebGLRunner } from "./backend/webgl-runner";
import type { CPURunner } from "./backend/cpu-runner";
import type { GPUArray } from "./array";

export type Shape = number[];
export type VectorShape<N extends number = number> = [N];
export type MatrixShape<R extends number = number, C extends number = number> = [R, C];

export interface ArrowImportOptions<S extends Shape = Shape> {
  shape?: S;
}

export interface BufferImportOptions<S extends Shape = Shape> {
  shape?: S;
  byteOffset?: number;
  length?: number;
}

/** Profiling entry for a single op */
export interface ProfilingEntry {
  op: string;
  durationMs: number;
  timestamp: number;
}

/**
 * Context returned by {@link init}. Provides array creation and canvas helpers.
 */
export interface AccelContext {
  /** @internal */
  backend: WebGPUBackend | WebGLBackend | CPUBackend;
  /** @internal */
  runner: KernelRunner | WebGLRunner | CPURunner;
  /** Backend in use: 'webgpu' | 'webgl' | 'cpu' */
  backendType: "webgpu" | "webgl" | "cpu";
  /** Whether Web Worker execution is active for CPU-heavy ops. */
  workerEnabled?: boolean;
  /** CPU execution engine when backendType is 'cpu'. */
  cpuEngine?: "js" | "wasm";
  /** Enable profiling. When true, recordOp stores entries. */
  enableProfiling(enable: boolean): void;
  /** Manually record an op (e.g. after timing). Only works when profiling enabled. */
  recordOp(op: string, durationMs: number): void;
  /** Get profiling results (op name, duration ms). */
  getProfilingResults(): ProfilingEntry[];
  /** Create a GPU-backed array from data. Optionally specify shape (e.g. [2, 3] for 2×3). */
  array<S extends Shape = Shape>(data: Float32Array | number[], shape?: S): GPUArray<S>;
  /** Create array of zeros. */
  zeros<S extends Shape = Shape>(shape: S): GPUArray<S>;
  /** Create array of ones. */
  ones<S extends Shape = Shape>(shape: S): GPUArray<S>;
  /** Create array filled with value. */
  full<S extends Shape = Shape>(shape: S, value: number): GPUArray<S>;
  /** Create array [start, start+step, ...] up to (not including) stop. */
  arange(start: number, stop: number, step?: number): GPUArray;
  /** Create array of num linearly spaced values from start to stop. */
  linspace(start: number, stop: number, num: number): GPUArray;
  /** Create array with uniform random values in [0, 1). */
  random(shape: number[]): GPUArray;
  /** Create array with standard normal (Gaussian) random values. */
  randn(shape: number[]): GPUArray;
  /** Create a GPU-backed array from ImageData (RGBA, normalized to [0,1]). */
  fromImageData(imageData: ImageData): GPUArray;
  /** Create a GPU-backed array from Apache Arrow-like column/vector data. */
  fromArrow<S extends Shape = Shape>(column: unknown, options?: ArrowImportOptions<S>): GPUArray<S>;
  /** Create a GPU-backed array from ArrayBuffer/SharedArrayBuffer binary Float32 data. */
  fromBuffer<S extends Shape = Shape>(
    buffer: ArrayBuffer | SharedArrayBuffer,
    options?: BufferImportOptions<S>
  ): GPUArray<S>;
  /** Render a GPU array to an HTMLCanvasElement (width×height). */
  toCanvas(arr: GPUArray, width: number, height: number): Promise<HTMLCanvasElement>;
  /** Run work in a deterministic disposal scope. Arrays created inside are disposed on scope exit. */
  scoped<T>(fn: (ctx: AccelContext) => Promise<T> | T): Promise<T>;
  /** Alias for scoped() inspired by TensorFlow.js tidy semantics. */
  tidy<T>(fn: (ctx: AccelContext) => Promise<T> | T): Promise<T>;
}

/**
 * GPUArray - GPU/CPU-backed array with NumPy-like API
 */

import type { WebGPUBackend } from "./backend/webgpu";
import type { CPUBackend } from "./backend/cpu-backend";
import type { KernelRunner } from "./backend/kernel-runner";
import type { CPURunner } from "./backend/cpu-runner";
import { errLengthMismatch } from "./errors";

const WORKGROUP_SIZE = 256;

type Backend = WebGPUBackend | CPUBackend;
type Runner = KernelRunner | CPURunner;
type Buffer = GPUBuffer | { data: Float32Array };

function isWebGPU(backend: Backend): backend is WebGPUBackend {
  return "device" in backend && backend.device !== undefined;
}

export class GPUArray {
  private backend: Backend;
  private runner: Runner;
  private buffer: Buffer;
  readonly length: number;
  readonly byteLength: number;
  private _shape: number[];

  constructor(
    backend: Backend,
    runner: Runner,
    buffer: Buffer,
    length: number,
    shape?: number[]
  ) {
    this.backend = backend;
    this.runner = runner;
    this.buffer = buffer;
    this.length = length;
    this.byteLength = length * 4;
    this._shape = shape ?? [length];
  }

  get shape(): number[] {
    return [...this._shape];
  }

  reshape(...dims: number[]): GPUArray {
    const total = dims.reduce((a, b) => a * b, 1);
    if (total !== this.length) {
      throw new Error(
        `reshape: cannot reshape [${this._shape.join(", ")}] (${this.length} elements) to [${dims.join(", ")}] (${total} elements).`
      );
    }
    this._shape = dims;
    return this;
  }

  async toArray(): Promise<Float32Array> {
    const result = new Float32Array(this.length);
    if (isWebGPU(this.backend)) {
      await this.backend.readBuffer(this.buffer as GPUBuffer, result.buffer);
    } else {
      (this.backend as CPUBackend).readBuffer(this.buffer as { data: Float32Array }, result.buffer);
    }
    return result;
  }

  async add(other: GPUArray | number): Promise<GPUArray> {
    if (typeof other === "number") {
      const scalarData = new Float32Array(this.length).fill(other);
      if (isWebGPU(this.backend)) {
        const scalarBuf = this.backend.createBufferFromData(
          scalarData.buffer as ArrayBuffer,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        const out = this.backend.createBuffer(
          this.byteLength,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        await (this.runner as KernelRunner).add(this.buffer as GPUBuffer, scalarBuf, out, this.length);
        this.backend.destroyBuffer(scalarBuf);
        this.backend.destroyBuffer(this.buffer as GPUBuffer);
        this.buffer = out;
      } else {
        const cpuRunner = this.runner as CPURunner;
        const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
        cpuRunner.add(this.buffer as { data: Float32Array }, { data: scalarData }, out, this.length);
        (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
        this.buffer = out;
      }
      return this;
    }
    if (other.length !== this.length) errLengthMismatch("add", this.length, other.length);

    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(
        this.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      await (this.runner as KernelRunner).add(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        out,
        this.length
      );
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
      this.buffer = out;
    } else {
      const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
      (this.runner as CPURunner).add(
        this.buffer as { data: Float32Array },
        other.buffer as { data: Float32Array },
        out,
        this.length
      );
      (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
      this.buffer = out;
    }
    return this;
  }

  async mul(other: GPUArray | number): Promise<GPUArray> {
    if (typeof other === "number") {
      if (isWebGPU(this.backend)) {
        const out = this.backend.createBuffer(
          this.byteLength,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        await (this.runner as KernelRunner).mulScalar(
          this.buffer as GPUBuffer,
          other,
          out,
          this.length
        );
        this.backend.destroyBuffer(this.buffer as GPUBuffer);
        this.buffer = out;
      } else {
        const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
        (this.runner as CPURunner).mulScalar(
          this.buffer as { data: Float32Array },
          other,
          out,
          this.length
        );
        (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
        this.buffer = out;
      }
      return this;
    }
    if (other.length !== this.length) errLengthMismatch("mul", this.length, other.length);

    if (isWebGPU(this.backend)) {
      const out = this.backend.createBuffer(
        this.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      await (this.runner as KernelRunner).mul(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        out,
        this.length
      );
      this.backend.destroyBuffer(this.buffer as GPUBuffer);
      this.buffer = out;
    } else {
      const out = (this.backend as CPUBackend).createBuffer(this.byteLength);
      (this.runner as CPURunner).mul(
        this.buffer as { data: Float32Array },
        other.buffer as { data: Float32Array },
        out,
        this.length
      );
      (this.backend as CPUBackend).destroyBuffer(this.buffer as { data: Float32Array });
      this.buffer = out;
    }
    return this;
  }

  async sum(): Promise<number> {
    if (this.length === 0) return 0;
    if (this.length === 1) {
      const result = new Float32Array(1);
      if (isWebGPU(this.backend)) {
        await this.backend.readBuffer(this.buffer as GPUBuffer, result.buffer);
      } else {
        (this.backend as CPUBackend).readBuffer(this.buffer as { data: Float32Array }, result.buffer);
      }
      return result[0];
    }

    if (isWebGPU(this.backend)) {
      let inputBuffer = this.buffer as GPUBuffer;
      let inputLength = this.length;
      while (true) {
        const outputLength = Math.ceil(inputLength / WORKGROUP_SIZE);
        const outputBuffer = this.backend.createBuffer(
          Math.max(4, outputLength * 4),
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        await (this.runner as KernelRunner).reduceSum(inputBuffer, outputBuffer, inputLength);
        if (inputBuffer !== this.buffer) this.backend.destroyBuffer(inputBuffer);
        if (outputLength === 1) {
          const result = new Float32Array(1);
          await this.backend.readBuffer(outputBuffer, result.buffer);
          this.backend.destroyBuffer(outputBuffer);
          return result[0];
        }
        inputBuffer = outputBuffer;
        inputLength = outputLength;
      }
    } else {
      const out = (this.backend as CPUBackend).createBuffer(4);
      (this.runner as CPURunner).reduceSum(
        this.buffer as { data: Float32Array },
        out,
        this.length
      );
      return out.data[0];
    }
  }

  async max(): Promise<number> {
    if (this.length === 0) return -Infinity;
    if (this.length === 1) {
      const result = new Float32Array(1);
      if (isWebGPU(this.backend)) {
        await this.backend.readBuffer(this.buffer as GPUBuffer, result.buffer);
      } else {
        (this.backend as CPUBackend).readBuffer(this.buffer as { data: Float32Array }, result.buffer);
      }
      return result[0];
    }

    if (isWebGPU(this.backend)) {
      let inputBuffer = this.buffer as GPUBuffer;
      let inputLength = this.length;
      while (true) {
        const outputLength = Math.ceil(inputLength / WORKGROUP_SIZE);
        const outputBuffer = this.backend.createBuffer(
          Math.max(4, outputLength * 4),
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        await (this.runner as KernelRunner).reduceMax(inputBuffer, outputBuffer, inputLength);
        if (inputBuffer !== this.buffer) this.backend.destroyBuffer(inputBuffer);
        if (outputLength === 1) {
          const result = new Float32Array(1);
          await this.backend.readBuffer(outputBuffer, result.buffer);
          this.backend.destroyBuffer(outputBuffer);
          return result[0];
        }
        inputBuffer = outputBuffer;
        inputLength = outputLength;
      }
    } else {
      const out = (this.backend as CPUBackend).createBuffer(4);
      (this.runner as CPURunner).reduceMax(
        this.buffer as { data: Float32Array },
        out,
        this.length
      );
      return out.data[0];
    }
  }

  async dot(other: GPUArray): Promise<number> {
    if (other.length !== this.length) errLengthMismatch("dot", this.length, other.length);
    if (isWebGPU(this.backend)) {
      const multiplied = this.backend.device!.createBuffer({
        size: this.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      await (this.runner as KernelRunner).mul(
        this.buffer as GPUBuffer,
        other.buffer as GPUBuffer,
        multiplied,
        this.length
      );
      const temp = new GPUArray(
        this.backend,
        this.runner,
        multiplied,
        this.length
      );
      const result = await temp.sum();
      this.backend.destroyBuffer(multiplied);
      return result;
    } else {
      let sum = 0;
      const a = (this.buffer as { data: Float32Array }).data;
      const b = (other.buffer as { data: Float32Array }).data;
      for (let i = 0; i < this.length; i++) sum += a[i] * b[i];
      return sum;
    }
  }

  getBuffer(): GPUBuffer | { data: Float32Array } {
    return this.buffer;
  }
}

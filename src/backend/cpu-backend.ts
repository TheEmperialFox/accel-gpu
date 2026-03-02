/**
 * CPU backend - for environments without WebGPU (Node, headless, fallback)
 */

import type { CPUBuffer } from "./cpu-runner";

export interface CPUBackend {
  type: "cpu";
  createBuffer(size: number, _usage?: number): CPUBuffer;
  createBufferFromData(data: ArrayBuffer, _usage?: number): CPUBuffer;
  writeBuffer(buffer: CPUBuffer, data: ArrayBuffer): void;
  readBuffer(buffer: CPUBuffer, output: ArrayBuffer): void;
  destroyBuffer(buffer: CPUBuffer): void;
}

export function createCPUBackend(): CPUBackend {
  return {
    type: "cpu",
    createBuffer(size: number, _usage?: number): CPUBuffer {
      return { data: new Float32Array(size / 4) };
    },
    createBufferFromData(data: ArrayBuffer, _usage?: number): CPUBuffer {
      return { data: new Float32Array(data) };
    },
    writeBuffer(buffer: CPUBuffer, data: ArrayBuffer): void {
      buffer.data.set(new Float32Array(data));
    },
    readBuffer(buffer: CPUBuffer, output: ArrayBuffer): void {
      new Float32Array(output).set(buffer.data);
    },
    destroyBuffer(_buffer: CPUBuffer): void {
      // No-op for CPU
    },
  };
}

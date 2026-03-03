import type { AccelContext, BufferImportOptions } from "../types";
import type { GPUArray } from "../array";

export function fromBuffer(
  ctx: AccelContext,
  buffer: ArrayBuffer | SharedArrayBuffer,
  options?: BufferImportOptions
): GPUArray {
  const byteOffset = options?.byteOffset ?? 0;
  if (byteOffset < 0) {
    throw new Error("fromBuffer: byteOffset must be >= 0");
  }
  if (byteOffset % 4 !== 0) {
    throw new Error("fromBuffer: byteOffset must be aligned to 4 bytes for Float32 data");
  }
  const availableBytes = buffer.byteLength - byteOffset;
  if (availableBytes < 0) {
    throw new Error("fromBuffer: byteOffset is out of bounds");
  }
  if (availableBytes % 4 !== 0 && options?.length === undefined) {
    throw new Error("fromBuffer: buffer length must be a multiple of 4 bytes for Float32 data");
  }

  const length = options?.length ?? availableBytes / 4;
  if (length < 0) {
    throw new Error("fromBuffer: length must be >= 0");
  }
  if (length * 4 > availableBytes) {
    throw new Error("fromBuffer: length exceeds available buffer bytes");
  }
  const view = new Float32Array(buffer, byteOffset, length);
  return ctx.array(view, options?.shape ?? [view.length]);
}

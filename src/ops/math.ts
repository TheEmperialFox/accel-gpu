/**
 * Math operations - add, mul, sum, max
 */

import type { GPUArray } from "../array";
import type { AccelContext } from "../types";

export function add(ctx: AccelContext, a: GPUArray, b: GPUArray): Promise<GPUArray> {
  return a.add(b);
}

export function mul(ctx: AccelContext, a: GPUArray, b: GPUArray): Promise<GPUArray> {
  return a.mul(b);
}

export function sum(ctx: AccelContext, a: GPUArray): Promise<number> {
  return a.sum();
}

export function max(ctx: AccelContext, a: GPUArray): Promise<number> {
  return a.max();
}

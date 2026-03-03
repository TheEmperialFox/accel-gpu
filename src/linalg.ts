export { GPUArray, init } from "./index";
export type { AccelContext, ArrowImportOptions, ProfilingEntry } from "./types";
export { matmul, dot, transpose } from "./ops/linear";
export { inv, det, solve, qr, svd } from "./ops/matrix";

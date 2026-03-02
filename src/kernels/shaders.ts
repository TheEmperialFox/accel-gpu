/**
 * Embedded WGSL compute shaders - zero runtime file loading
 */

export const ADD_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = a[i] + b[i];
  }
}
`;

export const MUL_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = a[i] * b[i];
  }
}
`;

export const MUL_SCALAR_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<uniform> scalar: f32;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&out)) {
    out[i] = a[i] * scalar;
  }
}
`;

export const REDUCE_SUM_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let idx = gid.x;
  let localIdx = lid.x;
  
  if (idx < arrayLength(&input)) {
    shared[localIdx] = input[idx];
  } else {
    shared[localIdx] = 0.0;
  }
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (localIdx < stride) {
      shared[localIdx] = shared[localIdx] + shared[localIdx + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
    if (stride == 0u) {
      break;
    }
  }

  if (localIdx == 0u) {
    output[wid.x] = shared[0];
  }
}
`;

export const REDUCE_MAX_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let idx = gid.x;
  let localIdx = lid.x;
  
  if (idx < arrayLength(&input)) {
    shared[localIdx] = input[idx];
  } else {
    shared[localIdx] = -1e38;
  }
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (localIdx < stride) {
      shared[localIdx] = max(shared[localIdx], shared[localIdx + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
    if (stride == 0u) {
      break;
    }
  }

  if (localIdx == 0u) {
    output[wid.x] = shared[0];
  }
}
`;

export const MATMUL_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: vec3<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let M = params.x;
  let N = params.y;
  let K = params.z;
  
  let i = gid.x;
  let j = gid.y;
  
  if (i < M && j < N) {
    var sum = 0.0;
    for (var k = 0u; k < K; k++) {
      sum += a[i * K + k] * b[k * N + j];
    }
    out[i * N + j] = sum;
  }
}
`;

export const SOFTMAX_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rows = params.x;
  let cols = params.y;
  let row = gid.x;
  
  if (row >= rows) {
    return;
  }
  
  var maxVal = -1e38;
  for (var c = 0u; c < cols; c++) {
    maxVal = max(maxVal, input[row * cols + c]);
  }
  
  var sumExp = 0.0;
  for (var c = 0u; c < cols; c++) {
    let e = exp(input[row * cols + c] - maxVal);
    output[row * cols + c] = e;
    sumExp += e;
  }
  
  for (var c = 0u; c < cols; c++) {
    output[row * cols + c] = output[row * cols + c] / sumExp;
  }
}
`;

export const LAYER_NORM_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: vec2<u32>;
let eps = 1e-5;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rows = params.x;
  let cols = params.y;
  let row = gid.x;
  
  if (row >= rows) { return; }
  
  var sum = 0.0;
  for (var c = 0u; c < cols; c++) {
    sum += input[row * cols + c];
  }
  let mean = sum / f32(cols);
  
  var varSum = 0.0;
  for (var c = 0u; c < cols; c++) {
    let d = input[row * cols + c] - mean;
    varSum += d * d;
  }
  let variance = sqrt(varSum / f32(cols) + eps);
  
  for (var c = 0u; c < cols; c++) {
    let normalized = (input[row * cols + c] - mean) / variance;
    output[row * cols + c] = normalized * gamma[c] + beta[c];
  }
}
`;

export const ATTENTION_SCORES_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec3<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let seq = params.x;
  let dim = params.y;
  let scale = 1.0 / sqrt(f32(dim));
  let i = gid.x;
  let j = gid.y;
  
  if (i >= seq || j >= seq) { return; }
  
  var score = 0.0;
  for (var d = 0u; d < dim; d++) {
    score += Q[i * dim + d] * K[j * dim + d];
  }
  output[i * seq + j] = score * scale;
}
`;

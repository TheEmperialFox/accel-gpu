// Softmax over last dimension. One thread per row (works for rows up to workgroup limit).
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>; // rows, cols

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rows = params.x;
  let cols = params.y;
  let row = gid.x;
  
  if (row >= rows) {
    return;
  }
  
  // Find max in row (for numerical stability)
  var maxVal = -1e38;
  for (var c = 0u; c < cols; c++) {
    maxVal = max(maxVal, input[row * cols + c]);
  }
  
  // Compute exp(x - max) and sum
  var sumExp = 0.0;
  for (var c = 0u; c < cols; c++) {
    let e = exp(input[row * cols + c] - maxVal);
    output[row * cols + c] = e;
    sumExp += e;
  }
  
  // Normalize
  for (var c = 0u; c < cols; c++) {
    output[row * cols + c] = output[row * cols + c] / sumExp;
  }
}

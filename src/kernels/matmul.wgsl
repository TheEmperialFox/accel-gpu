@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: vec3<u32>; // M, N, K

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

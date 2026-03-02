// Two-phase reduction: phase 1 reduces within workgroups, phase 2 reduces group results
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
  
  // Load into shared memory (or 0 if out of bounds)
  if (idx < arrayLength(&input)) {
    shared[localIdx] = input[idx];
  } else {
    shared[localIdx] = 0.0;
  }
  workgroupBarrier();

  // Tree reduction within workgroup
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

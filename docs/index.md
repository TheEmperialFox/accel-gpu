---
layout: home

hero:
  name: "accel-gpu"
  text: "NumPy for the browser GPU"
  tagline: "WebGPU-first math with automatic WebGL2/CPU fallback."
  actions:
    - theme: brand
      text: Quick Start
      link: /guide/quickstart
    - theme: alt
      text: Explore Demos
      link: /demos
    - theme: alt
      text: Open Playground
      link: /playground

features:
  - title: Zero-shader API
    details: NumPy-style operations without writing WGSL.
  - title: Smart fallback chain
    details: WebGPU → WebGL2 → CPU for broad browser compatibility.
  - title: Tree-shakeable modules
    details: Import only what you need via math/linalg/ml/signal/data subpaths.
  - title: Data interop
    details: Arrow-like ingestion and raw ArrayBuffer/SharedArrayBuffer support.
  - title: Memory safety
    details: Use gpu.scoped/gpu.tidy with FinalizationRegistry fallback cleanup.
  - title: Browser-tested
    details: Cross-browser Playwright coverage on Chromium, Firefox, and WebKit.
---

## Try It Live

<iframe src="https://phantasm0009.github.io/accel-gpu/playground/" width="100%" height="560"></iframe>

## Examples

- [Demos Hub](/demos)
- [Image Processing Demo](/demos/image)
- [Heatmap Demo](/demos/heatmap)
- [Neural Network Demo](/demos/nn)
- [N-Body Demo](/demos/nbody)
- [Local Audio Demo](/demos/audio)
- [Vector Search Demo](/demos/vector-search)

import { defineConfig } from "vitepress";

export default defineConfig({
  title: "accel-gpu",
  description: "NumPy for the browser GPU",
  base: "/accel-gpu/",
  head: [["link", { rel: "icon", href: "/accel-gpu/icon.png" }]],
  themeConfig: {
    nav: [
      { text: "Quick Start", link: "/guide/quickstart" },
      { text: "API", link: "/api" },
      { text: "Demos", link: "/demos" },
      { text: "Playground", link: "/playground" },
      { text: "Benchmarks", link: "/benchmarks" },
    ],
    sidebar: [
      {
        text: "Getting Started",
        items: [
          { text: "Overview", link: "/" },
          { text: "Quick Start", link: "/guide/quickstart" },
          { text: "Memory Management", link: "/guide/memory-management" },
          { text: "Backend Tolerance", link: "/guide/backend-tolerance" },
        ],
      },
      {
        text: "Reference",
        items: [
          { text: "API Reference", link: "/api" },
        ],
      },
      {
        text: "Explore",
        items: [
          { text: "Demos Hub", link: "/demos" },
          { text: "Demo: Core", link: "/demos/core" },
          { text: "Demo: Image", link: "/demos/image" },
          { text: "Demo: Heatmap", link: "/demos/heatmap" },
          { text: "Demo: NN", link: "/demos/nn" },
          { text: "Demo: N-Body", link: "/demos/nbody" },
          { text: "Demo: Audio", link: "/demos/audio" },
          { text: "Demo: Vector Search", link: "/demos/vector-search" },
          { text: "Playground", link: "/playground" },
          { text: "Benchmarks", link: "/benchmarks" },
        ],
      },
    ],
    search: {
      provider: "local",
    },
    socialLinks: [{ icon: "github", link: "https://github.com/Phantasm0009/accel-gpu" }],
    footer: {
      message: "MIT Licensed",
      copyright: "Copyright © 2026 accel-gpu contributors",
    },
    outline: "deep",
  },
});

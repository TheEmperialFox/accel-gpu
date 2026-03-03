import { defineConfig } from "vitepress";

export default defineConfig({
  title: "accel-gpu",
  description: "NumPy for the browser GPU",
  base: "/accel-gpu/",
  themeConfig: {
    nav: [
      { text: "Guide", link: "/guide/quickstart" },
      { text: "API", link: "/api" },
      { text: "Playground", link: "https://phantasm0009.github.io/accel-gpu/playground/" },
    ],
    sidebar: [
      {
        text: "Docs",
        items: [
          { text: "Overview", link: "/" },
          { text: "Quick Start", link: "/guide/quickstart" },
          { text: "API Reference", link: "/api" },
        ],
      },
    ],
  },
});

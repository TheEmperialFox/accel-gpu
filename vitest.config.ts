import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    exclude: ["playwright/**", "node_modules/**", "dist/**"],
  },
});

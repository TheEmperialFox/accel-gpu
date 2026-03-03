import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/math.ts", "src/linalg.ts", "src/ml.ts", "src/signal.ts", "src/data.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  splitting: false,
  treeshake: true,
  minify: false,
  target: "es2020",
  outDir: "dist",
});

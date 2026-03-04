import { stat } from "node:fs/promises";
import { resolve } from "node:path";

const limitsKb = {
  "dist/index.js": 260,
  "dist/index.cjs": 260,
  "dist/math.js": 220,
  "dist/linalg.js": 220,
  "dist/ml.js": 220,
  "dist/signal.js": 220,
  "dist/data.js": 220,
};

async function main() {
  const failures = [];

  for (const [relativePath, maxKb] of Object.entries(limitsKb)) {
    const fullPath = resolve(process.cwd(), relativePath);
    const info = await stat(fullPath);
    const sizeKb = info.size / 1024;
    if (sizeKb > maxKb) {
      failures.push(`${relativePath}: ${sizeKb.toFixed(1)}KB > ${maxKb}KB`);
    } else {
      console.log(`${relativePath}: ${sizeKb.toFixed(1)}KB <= ${maxKb}KB`);
    }
  }

  if (failures.length > 0) {
    console.error("Bundle size checks failed:");
    for (const failure of failures) {
      console.error(`- ${failure}`);
    }
    process.exit(1);
  }

  console.log("Bundle size checks passed.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

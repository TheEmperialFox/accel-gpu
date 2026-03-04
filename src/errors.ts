/**
 * Clear, descriptive error messages. Stack traces are included automatically when thrown.
 */

export function errLengthMismatch(op: string, a: number, b: number): never {
  throw new Error(`${op}: length mismatch — ${a} vs ${b}. Arrays must have the same length.`);
}

export function errShapeMismatch(op: string, expected: string, actual: string): never {
  throw new Error(`${op}: shape mismatch — expected ${expected}, got ${actual}.`);
}

export function errMatmulShapes(op: string, aShape: string, bShape: string, reason: string): never {
  throw new Error(`${op}: A is ${aShape}, B is ${bShape} — ${reason}`);
}

export function errInvalidShape(op: string, shape: number[], expectedElements: number): never {
  throw new Error(
    `${op}: shape [${shape.join(", ")}] has ${shape.reduce((a, b) => a * b, 1)} elements, expected ${expectedElements}.`
  );
}

export function errRequiresRank(op: string, expected: string, actualRank: number): never {
  throw new Error(`${op}: expected ${expected}, got rank ${actualRank}.`);
}

export function errRequires2DOrExplicit(op: string): never {
  throw new Error(`${op}: provide rows and cols, or use a 2D array shape.`);
}

export function errRequiresSquare(op: string, rows: number, cols: number): never {
  throw new Error(`${op}: matrix must be square, got ${rows}×${cols}.`);
}

export function errPowerOfTwo(op: string, valueName: string, value: number): never {
  throw new Error(`${op}: ${valueName} must be a power of 2, got ${value}.`);
}

export function errEvenLength(op: string, length: number, details: string): never {
  throw new Error(`${op}: length ${length} is invalid — ${details}`);
}

/**
 * WebGL2 backend - render-to-texture compute for environments without WebGPU
 * Uses R16F (half-float) for compatibility - R32F requires EXT_color_buffer_float
 * which Safari and some browsers don't support.
 */

function float32ToFloat16(f32: number): number {
  const f32v = new Float32Array([f32]);
  const u32 = new Uint32Array(f32v.buffer)[0];
  const sign = (u32 >> 16) & 0x8000;
  let exp = (u32 >> 23) & 0xff;
  const mant = u32 & 0x7fffff;
  if (exp === 255) return sign | 0x7c00;
  if (exp === 0 && mant === 0) return sign;
  exp = exp - 127 + 15;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) return sign;
  return sign | (exp << 10) | (mant >> 13);
}

function float16ToFloat32(h16: number): number {
  const sign = (h16 & 0x8000) >> 15;
  const exp = (h16 & 0x7c00) >> 10;
  const frac = h16 & 0x3ff;
  if (exp === 0 && frac === 0) return sign ? -0 : 0;
  if (exp === 31) return frac ? NaN : (sign ? -Infinity : Infinity);
  const m = exp === 0 ? frac / 1024 : 1 + frac / 1024;
  const e = exp === 0 ? -14 : exp - 15;
  return (sign ? -1 : 1) * m * Math.pow(2, e);
}

export interface WebGLBuffer {
  texture: WebGLTexture;
  width: number;
  height: number;
  length: number;
}

function getTextureDimensions(length: number): { width: number; height: number } {
  const pixels = Math.max(1, length);
  const maxSize = 4096;
  let width = Math.ceil(Math.sqrt(pixels));
  let height = Math.ceil(pixels / width);
  if (width > maxSize) width = maxSize;
  if (height > maxSize) height = maxSize;
  return { width, height };
}

export interface WebGLBackend {
  type: "webgl";
  gl: WebGL2RenderingContext;
  canvas: HTMLCanvasElement;
  createBuffer(size: number, _usage?: number): WebGLBuffer;
  createBufferFromData(data: ArrayBuffer, _usage?: number): WebGLBuffer;
  writeBuffer(buffer: WebGLBuffer, data: ArrayBuffer): void;
  readBuffer(buffer: WebGLBuffer, output: ArrayBuffer): Promise<void>;
  destroyBuffer(buffer: WebGLBuffer): void;
}

function testFramebufferCompatibility(gl: WebGL2RenderingContext): boolean {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, 2, 2, 0, gl.RED, gl.HALF_FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(fb);
  gl.deleteTexture(tex);
  return status === gl.FRAMEBUFFER_COMPLETE;
}

export function createWebGLBackend(): WebGLBackend {
  const canvas = document.createElement("canvas");
  canvas.width = 4096;
  canvas.height = 4096;

  const glOrNull = canvas.getContext("webgl2", {
    preserveDrawingBuffer: true,
    antialias: false,
    depth: false,
    stencil: false,
  });

  if (!glOrNull) {
    throw new Error("WebGL2 is not supported");
  }

  const gl = glOrNull;

  if (!testFramebufferCompatibility(gl)) {
    throw new Error("WebGL2 R16F framebuffer not supported (Safari/older GPUs)");
  }

  const texturePool: WebGLBuffer[] = [];

  function createTexture(width: number, height: number, length: number): WebGLBuffer {
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, width, height, 0, gl.RED, gl.HALF_FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return { texture, width, height, length };
  }

  return {
    type: "webgl",
    gl,
    canvas,

    createBuffer(size: number, _usage?: number): WebGLBuffer {
      const length = size / 4;
      const { width, height } = getTextureDimensions(length);
      const reused = texturePool.find((b) => b.width === width && b.height === height && b.length >= length);
      if (reused) {
        texturePool.splice(texturePool.indexOf(reused), 1);
        reused.length = length;
        return reused;
      }
      return createTexture(width, height, length);
    },

    createBufferFromData(data: ArrayBuffer, _usage?: number): WebGLBuffer {
      const floats = new Float32Array(data);
      const length = floats.length;
      const { width, height } = getTextureDimensions(length);
      const buf = createTexture(width, height, length);
      const padded = new Uint16Array(width * height);
      for (let i = 0; i < length; i++) padded[i] = float32ToFloat16(floats[i]);
      gl.bindTexture(gl.TEXTURE_2D, buf.texture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RED, gl.HALF_FLOAT, padded);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return buf;
    },

    writeBuffer(buffer: WebGLBuffer, data: ArrayBuffer): void {
      const floats = new Float32Array(data);
      const half = new Uint16Array(buffer.length);
      for (let i = 0; i < buffer.length; i++) half[i] = float32ToFloat16(floats[i]);
      gl.bindTexture(gl.TEXTURE_2D, buffer.texture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, buffer.width, buffer.height, gl.RED, gl.HALF_FLOAT, half);
      gl.bindTexture(gl.TEXTURE_2D, null);
    },

    async readBuffer(buffer: WebGLBuffer, output: ArrayBuffer): Promise<void> {
      const fb = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, buffer.texture, 0);
      const half = new Uint16Array(buffer.length);
      gl.readPixels(0, 0, buffer.width, buffer.height, gl.RED, gl.HALF_FLOAT, half);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteFramebuffer(fb);
      const floats = new Float32Array(output.byteLength / 4);
      for (let i = 0; i < floats.length; i++) floats[i] = float16ToFloat32(half[i]);
      new Float32Array(output).set(floats);
    },

    destroyBuffer(buffer: WebGLBuffer): void {
      if (texturePool.length < 32) {
        texturePool.push(buffer);
      } else {
        gl.deleteTexture(buffer.texture);
      }
    },
  };
}

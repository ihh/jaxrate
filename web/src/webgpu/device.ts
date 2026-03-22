/**
 * WebGPU device management.
 * Provides lazy initialization and shader compilation utilities.
 */

let _device: GPUDevice | null = null;
let _adapter: GPUAdapter | null = null;

/** Get or initialize the WebGPU device. */
export async function getDevice(): Promise<GPUDevice> {
  if (_device) return _device;

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU not available in this environment');
  }

  _adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!_adapter) throw new Error('No WebGPU adapter found');

  _device = await _adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: _adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: _adapter.limits.maxBufferSize,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
    },
  });

  _device.lost.then((info) => {
    console.error('WebGPU device lost:', info.message);
    _device = null;
  });

  return _device;
}

/** Release the WebGPU device. */
export function releaseDevice(): void {
  if (_device) {
    _device.destroy();
    _device = null;
    _adapter = null;
  }
}

/** Create a storage buffer with initial data. */
export function createStorageBuffer(
  device: GPUDevice, data: Float32Array | Int32Array | Uint32Array,
  usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage,
    mappedAtCreation: true,
  });
  if (data instanceof Float32Array) {
    new Float32Array(buffer.getMappedRange()).set(data);
  } else if (data instanceof Int32Array) {
    new Int32Array(buffer.getMappedRange()).set(data);
  } else {
    new Uint32Array(buffer.getMappedRange()).set(data);
  }
  buffer.unmap();
  return buffer;
}

/** Create an empty storage buffer. */
export function createEmptyBuffer(
  device: GPUDevice, size: number,
  usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
): GPUBuffer {
  return device.createBuffer({ size, usage });
}

/** Read back a GPU buffer to CPU. */
export async function readBuffer(
  device: GPUDevice, buffer: GPUBuffer, size: number
): Promise<Float32Array> {
  const readback = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return result;
}

/** Compile a compute shader module. */
export function createShaderModule(device: GPUDevice, code: string): GPUShaderModule {
  return device.createShaderModule({ code });
}

/** Create a compute pipeline with bind group layout. */
export function createComputePipeline(
  device: GPUDevice, module: GPUShaderModule, entryPoint: string,
  layout: GPUBindGroupLayout
): GPUComputePipeline {
  return device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint },
  });
}

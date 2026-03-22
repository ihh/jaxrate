/**
 * WebGPU-accelerated HMM algorithms.
 * Diagonal sweep: columns processed sequentially, states in parallel via compute shaders.
 *
 * For K states and C columns:
 *   - CPU: O(CK²) sequential
 *   - GPU: O(C × K/W × log(W)) where W = workgroup size (256)
 *   - Advantage: ~K/256 speedup for large state spaces (K > 256)
 */

import type { TerminalWeights, ParseResult } from '../types.js';
import type { HmmTables } from '../hmm.js';
import { NEG_INF, logsumexp } from '../log-semiring.js';
import {
  getDevice, createStorageBuffer, createEmptyBuffer,
  readBuffer, createShaderModule,
} from './device.js';

// Inline shader source — bundled at build time or loaded from files
import { FORWARD_SHADER, BACKWARD_SHADER, VITERBI_SHADER } from './shader-source.js';

// ── GPU resource cache ──────────────────────────────────────────────

interface GpuHmmResources {
  device: GPUDevice;
  K: number;
  C: number;
  // Buffers
  transBuffer: GPUBuffer;
  emitBuffer: GPUBuffer;
  alphaBuffers: [GPUBuffer, GPUBuffer];  // ping-pong
  paramBuffer: GPUBuffer;
  // Pipelines
  forwardPipeline: GPUComputePipeline;
  forwardBindGroupLayout: GPUBindGroupLayout;
  backwardPipeline: GPUComputePipeline;
  backwardBindGroupLayout: GPUBindGroupLayout;
  viterbiPipeline: GPUComputePipeline;
  viterbiBindGroupLayout: GPUBindGroupLayout;
}

let _resources: GpuHmmResources | null = null;

// ── Float64 → Float32 conversion ────────────────────────────────────

function f64ToF32(arr: Float64Array): Float32Array {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i];
  return out;
}

function twToF32(tw: TerminalWeights): { single: Float32Array[]; paired: Float32Array[] | null } {
  return {
    single: tw.single.map(f64ToF32),
    paired: tw.paired ? tw.paired.map(f64ToF32) : null,
  };
}

// ── Initialize GPU resources ────────────────────────────────────────

async function initResources(tables: HmmTables): Promise<GpuHmmResources> {
  const device = await getDevice();
  const { K, C } = tables;

  // Upload transition matrix (K×K) and emission matrix (K×C) as f32
  const transF32 = f64ToF32(tables.logTrans);
  const emitF32 = f64ToF32(tables.logEmit);
  const transBuffer = createStorageBuffer(device, transF32);
  const emitBuffer = createStorageBuffer(device, emitF32);

  // Ping-pong alpha buffers
  const alphaBuffers: [GPUBuffer, GPUBuffer] = [
    createEmptyBuffer(device, K * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
    createEmptyBuffer(device, K * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
  ];

  // Params uniform buffer (16 bytes: K, C, col, pad)
  const paramBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Compile shaders
  const forwardModule = createShaderModule(device, FORWARD_SHADER);
  const backwardModule = createShaderModule(device, BACKWARD_SHADER);
  const viterbiModule = createShaderModule(device, VITERBI_SHADER);

  // Forward bind group layout: trans, emit, alpha_prev, alpha_curr, params
  const forwardBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const forwardPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [forwardBindGroupLayout] }),
    compute: { module: forwardModule, entryPoint: 'forward_step' },
  });

  // Backward: same layout
  const backwardBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const backwardPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [backwardBindGroupLayout] }),
    compute: { module: backwardModule, entryPoint: 'backward_step' },
  });

  // Viterbi: trans, emit, v_prev, v_curr, bp_curr, params (6 bindings)
  const viterbiBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const viterbiPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [viterbiBindGroupLayout] }),
    compute: { module: viterbiModule, entryPoint: 'viterbi_step' },
  });

  const resources: GpuHmmResources = {
    device, K, C,
    transBuffer, emitBuffer, alphaBuffers, paramBuffer,
    forwardPipeline, forwardBindGroupLayout,
    backwardPipeline, backwardBindGroupLayout,
    viterbiPipeline, viterbiBindGroupLayout,
  };

  _resources = resources;
  return resources;
}

// ── GPU Forward ─────────────────────────────────────────────────────

export async function hmmForwardGpu(tables: HmmTables): Promise<{
  alpha: Float32Array;       // (C, K) — final alpha values on CPU
  logLikelihood: number;
}> {
  const res = await initResources(tables);
  const { device, K, C } = res;

  // Initialize alpha[0, k] = logInit[k] + logEmit[k, 0]
  const initAlpha = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    initAlpha[k] = tables.logInit[k] + tables.logEmit[k * C];
  }

  // Upload initial alpha to buffer 0
  device.queue.writeBuffer(res.alphaBuffers[0], 0, initAlpha);

  // Store all alpha values for CPU readback
  const allAlpha = new Float32Array(C * K);
  allAlpha.set(initAlpha, 0);

  // Process columns 1..C-1 with ping-pong
  for (let c = 1; c < C; c++) {
    const prevBuf = res.alphaBuffers[(c - 1) % 2];
    const currBuf = res.alphaBuffers[c % 2];

    // Update params
    const params = new Uint32Array([K, C, c, 0]);
    device.queue.writeBuffer(res.paramBuffer, 0, params);

    const bindGroup = device.createBindGroup({
      layout: res.forwardBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: res.transBuffer } },
        { binding: 1, resource: { buffer: res.emitBuffer } },
        { binding: 2, resource: { buffer: prevBuf } },
        { binding: 3, resource: { buffer: currBuf } },
        { binding: 4, resource: { buffer: res.paramBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(res.forwardPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(K);  // one workgroup per destination state
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Read back this column's alpha (for final output and termination)
    const colAlpha = await readBuffer(device, currBuf, K * 4);
    allAlpha.set(colAlpha, c * K);
  }

  // Termination: logLik = logsumexp(alpha[C-1, k] + logTerm[k])
  const tmp = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    tmp[k] = allAlpha[(C - 1) * K + k] + tables.logTerm[k];
  }
  const logLikelihood = logsumexp(tmp);

  return { alpha: allAlpha, logLikelihood };
}

// ── GPU Backward ────────────────────────────────────────────────────

export async function hmmBackwardGpu(tables: HmmTables): Promise<{
  beta: Float32Array;
  logLikelihood: number;
}> {
  const res = await initResources(tables);
  const { device, K, C } = res;

  // Initialize beta[C-1, k] = logTerm[k]
  const initBeta = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    initBeta[k] = tables.logTerm[k];
  }

  const allBeta = new Float32Array(C * K);
  allBeta.set(initBeta, (C - 1) * K);

  // Upload to buffer
  const betaBufIdx = (C - 1) % 2;
  device.queue.writeBuffer(res.alphaBuffers[betaBufIdx], 0, initBeta);

  // Process columns C-2 down to 0
  for (let c = C - 2; c >= 0; c--) {
    const nextBuf = res.alphaBuffers[(c + 1) % 2];
    const currBuf = res.alphaBuffers[c % 2];

    const params = new Uint32Array([K, C, c, c + 1]);
    device.queue.writeBuffer(res.paramBuffer, 0, params);

    const bindGroup = device.createBindGroup({
      layout: res.backwardBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: res.transBuffer } },
        { binding: 1, resource: { buffer: res.emitBuffer } },
        { binding: 2, resource: { buffer: nextBuf } },
        { binding: 3, resource: { buffer: currBuf } },
        { binding: 4, resource: { buffer: res.paramBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(res.backwardPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(K);
    pass.end();
    device.queue.submit([encoder.finish()]);

    const colBeta = await readBuffer(device, currBuf, K * 4);
    allBeta.set(colBeta, c * K);
  }

  // Termination
  const tmp = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    tmp[k] = tables.logInit[k] + tables.logEmit[k * C] + allBeta[k];
  }
  const logLikelihood = logsumexp(tmp);

  return { beta: allBeta, logLikelihood };
}

// ── GPU Viterbi ─────────────────────────────────────────────────────

export async function hmmViterbiGpu(tables: HmmTables): Promise<ParseResult> {
  const res = await initResources(tables);
  const { device, K, C } = res;

  // Initialize v[0, k] = logInit[k] + logEmit[k, 0]
  const initV = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    initV[k] = tables.logInit[k] + tables.logEmit[k * C];
  }

  device.queue.writeBuffer(res.alphaBuffers[0], 0, initV);

  // Backpointer storage: (C, K) on CPU
  const allBp = new Uint32Array(C * K);

  // Create backpointer buffer on GPU
  const bpBuffer = createEmptyBuffer(
    device, K * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );

  // Store all v values for traceback
  const allV = new Float32Array(C * K);
  allV.set(initV, 0);

  for (let c = 1; c < C; c++) {
    const prevBuf = res.alphaBuffers[(c - 1) % 2];
    const currBuf = res.alphaBuffers[c % 2];

    const params = new Uint32Array([K, C, c, 0]);
    device.queue.writeBuffer(res.paramBuffer, 0, params);

    const bindGroup = device.createBindGroup({
      layout: res.viterbiBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: res.transBuffer } },
        { binding: 1, resource: { buffer: res.emitBuffer } },
        { binding: 2, resource: { buffer: prevBuf } },
        { binding: 3, resource: { buffer: currBuf } },
        { binding: 4, resource: { buffer: bpBuffer } },
        { binding: 5, resource: { buffer: res.paramBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(res.viterbiPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(K);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Read back v and bp for this column
    const colV = await readBuffer(device, currBuf, K * 4);
    allV.set(colV, c * K);

    const bpReadback = device.createBuffer({
      size: K * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(bpBuffer, 0, bpReadback, 0, K * 4);
    device.queue.submit([enc2.finish()]);
    await bpReadback.mapAsync(GPUMapMode.READ);
    const bpData = new Uint32Array(bpReadback.getMappedRange().slice(0));
    bpReadback.unmap();
    bpReadback.destroy();
    allBp.set(bpData, c * K);
  }

  bpBuffer.destroy();

  // Termination: find best final state
  let bestFinal = -Infinity;
  let bestState = 0;
  for (let k = 0; k < K; k++) {
    const score = allV[(C - 1) * K + k] + tables.logTerm[k];
    if (score > bestFinal) {
      bestFinal = score;
      bestState = k;
    }
  }

  // Traceback on CPU
  const labels = new Int32Array(C);
  labels[C - 1] = bestState;
  for (let c = C - 2; c >= 0; c--) {
    labels[c] = allBp[(c + 1) * K + labels[c + 1]];
  }

  return { labels, logProb: bestFinal };
}

// ── GPU Posteriors ──────────────────────────────────────────────────

export async function hmmPosteriorsGpu(tables: HmmTables): Promise<{
  posteriors: Float32Array;  // (C, K)
  logLikelihood: number;
}> {
  const [fwd, bwd] = await Promise.all([
    hmmForwardGpu(tables),
    hmmBackwardGpu(tables),
  ]);

  const { K, C } = tables;
  const posteriors = new Float32Array(C * K);

  for (let c = 0; c < C; c++) {
    const tmp = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      tmp[k] = fwd.alpha[c * K + k] + bwd.beta[c * K + k];
    }
    const logZ = logsumexp(tmp);
    for (let k = 0; k < K; k++) {
      posteriors[c * K + k] = Math.exp(tmp[k] - logZ);
    }
  }

  return { posteriors, logLikelihood: fwd.logLikelihood };
}

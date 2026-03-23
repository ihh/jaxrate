/**
 * WebGPU-accelerated SCFG inside and Viterbi.
 *
 * Span-level wavefront: outer loop over span length (sequential),
 * inner dispatch over all (i, rule) pairs in parallel.
 *
 * For each span, three kernels execute in sequence:
 *   1. emit_step: terminal, left-emit, paired rules — 1 thread per position i
 *   2. binary_step: binary rules — 256 threads per (i, rule), split-point reduction
 *   3. unary_propagate: null unary chains — 1 thread per position i, multi-pass
 *
 * Complexity: O(C² × R + C × R_bin × C/256) per span = O(C³K²) total for binary-heavy grammars.
 * The split-point reduction gives ~C/256 speedup over CPU for the binary inner loop.
 */

import type { CompiledGrammar, TerminalWeights, ParseResult } from '../types.js';
import type { InsideResult } from '../scfg.js';
import { NEG_INF, negInfArray } from '../log-semiring.js';
import { classifyRules } from '../grammar.js';
import {
  getDevice, createStorageBuffer, createEmptyBuffer, readBuffer,
  createShaderModule,
} from './device.js';
import { SCFG_INSIDE_SHADER, SCFG_VITERBI_SHADER } from './shader-source.js';

// ── Rule packing ────────────────────────────────────────────────────
// Each rule → 6 u32s: [lhs, rhs0, rhs1, emit_model, emit_npos, bitcast(log_weight)]

function packRules(cg: CompiledGrammar, ruleIndices: number[]): Uint32Array {
  const packed = new Uint32Array(ruleIndices.length * 6);
  const f32Tmp = new Float32Array(1);
  const u32View = new Uint32Array(f32Tmp.buffer);

  for (let idx = 0; idx < ruleIndices.length; idx++) {
    const r = ruleIndices[idx];
    const base = idx * 6;
    packed[base + 0] = cg.ruleLhs[r];
    packed[base + 1] = cg.ruleNRhs[r] >= 1 ? cg.ruleRhs[r][0] : 0xFFFFFFFF;
    packed[base + 2] = cg.ruleNRhs[r] >= 2 ? cg.ruleRhs[r][1] : 0xFFFFFFFF;
    packed[base + 3] = cg.ruleNEmissions[r] > 0 ? cg.ruleEmissionModel[r][0] : 0xFFFFFFFF;
    packed[base + 4] = cg.ruleNEmissions[r] > 0 ? cg.ruleEmissionNPos[r][0] : 0;
    f32Tmp[0] = cg.ruleLogWeights[r];
    packed[base + 5] = u32View[0];
  }
  return packed;
}

function f64ToF32(arr: Float64Array): Float32Array {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i];
  return out;
}

// ── SCFG Inside (GPU) ──────────────────────────────────────────────

export async function scfgInsideGpu(
  cg: CompiledGrammar, tw: TerminalWeights
): Promise<InsideResult> {
  const device = await getDevice();
  const K = cg.K;
  const C = tw.C;
  const S = C + 1;
  const chartSize = K * S * S;

  const rules = classifyRules(cg);

  // Order: terminal, emitUnary, emitPaired, binary, unary
  const orderedRules = [
    ...rules.terminal, ...rules.emitUnary, ...rules.emitPaired,
    ...rules.binary, ...rules.unary,
  ];
  const packedRules = packRules(cg, orderedRules);

  // Count paired models
  const kPaired = tw.paired ? tw.paired.length : 0;

  // Upload buffers
  const alphaInit = new Float32Array(chartSize).fill(NEG_INF);

  // Apply epsilon rules on CPU (span 0)
  for (const r of rules.epsilon) {
    const lhs = cg.ruleLhs[r];
    const logW = cg.ruleLogWeights[r];
    for (let i = 0; i <= C; i++) {
      const idx = lhs * S * S + i * S + i;
      alphaInit[idx] = Math.max(alphaInit[idx], NEG_INF) === NEG_INF
        ? logW
        : logaddexpF32(alphaInit[idx], logW);
    }
  }

  // Propagate span-0 unary on CPU (simple, few iterations)
  for (let pass = 0; pass < K; pass++) {
    let changed = false;
    for (const r of rules.unary) {
      const lhs = cg.ruleLhs[r];
      const rhs = cg.ruleRhs[r][0];
      const logW = cg.ruleLogWeights[r];
      for (let i = 0; i <= C; i++) {
        const childIdx = rhs * S * S + i * S + i;
        const childVal = alphaInit[childIdx];
        if (childVal > NEG_INF + 1e30) {
          const idx = lhs * S * S + i * S + i;
          const newVal = logaddexpF32(alphaInit[idx], logW + childVal);
          if (newVal > alphaInit[idx] + 1e-6) {
            alphaInit[idx] = newVal;
            changed = true;
          }
        }
      }
    }
    if (!changed) break;
  }

  const alphaBuffer = createStorageBuffer(device, alphaInit,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

  // Single emission weights: flatten all models into one buffer
  const singleFlat = new Float32Array(tw.single.length * C);
  for (let m = 0; m < tw.single.length; m++) {
    for (let c = 0; c < C; c++) singleFlat[m * C + c] = tw.single[m][c];
  }
  const singleBuffer = createStorageBuffer(device, singleFlat);

  // Paired emission weights
  const pairedSize = kPaired > 0 ? kPaired * C * C : 4;  // min 4 bytes
  const pairedFlat = new Float32Array(pairedSize / 4 || 1);
  if (tw.paired) {
    for (let m = 0; m < kPaired; m++) {
      for (let c = 0; c < C * C; c++) pairedFlat[m * C * C + c] = tw.paired[m][c];
    }
  }
  const pairedBuffer = createStorageBuffer(device, pairedFlat);

  const rulesBuffer = createStorageBuffer(device, packedRules);

  // Params uniform (48 bytes = 12 u32s)
  const paramBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Compile shader
  const module = createShaderModule(device, SCFG_INSIDE_SHADER);

  // Bind group layout (5 bindings)
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  const emitPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: 'emit_step' },
  });
  const binaryPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: 'binary_step' },
  });
  const unaryPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: 'unary_propagate' },
  });

  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: alphaBuffer } },
      { binding: 1, resource: { buffer: singleBuffer } },
      { binding: 2, resource: { buffer: pairedBuffer } },
      { binding: 3, resource: { buffer: rulesBuffer } },
      { binding: 4, resource: { buffer: paramBuffer } },
    ],
  });

  // Span-level wavefront: dispatch per span
  for (let span = 1; span <= C; span++) {
    const nPositions = C - span + 1;

    // Update params
    const params = new Uint32Array([
      K, C, S, span,
      rules.terminal.length,
      rules.emitUnary.length,
      rules.emitPaired.length,
      rules.binary.length,
      rules.unary.length,
      kPaired,
      0, 0,
    ]);
    device.queue.writeBuffer(paramBuffer, 0, params);

    const encoder = device.createCommandEncoder();

    // Kernel 1: emit rules (1 workgroup per position)
    const pass1 = encoder.beginComputePass();
    pass1.setPipeline(emitPipeline);
    pass1.setBindGroup(0, bindGroup);
    pass1.dispatchWorkgroups(nPositions);
    pass1.end();

    // Kernel 2: binary rules (nPositions × nBinary workgroups, 256 threads each)
    if (rules.binary.length > 0) {
      const pass2 = encoder.beginComputePass();
      pass2.setPipeline(binaryPipeline);
      pass2.setBindGroup(0, bindGroup);
      pass2.dispatchWorkgroups(nPositions, rules.binary.length);
      pass2.end();
    }

    // Kernel 3: unary propagation (1 workgroup per position)
    if (rules.unary.length > 0) {
      const pass3 = encoder.beginComputePass();
      pass3.setPipeline(unaryPipeline);
      pass3.setBindGroup(0, bindGroup);
      pass3.dispatchWorkgroups(nPositions);
      pass3.end();
    }

    device.queue.submit([encoder.finish()]);

    // Sync between spans (GPU ensures kernel ordering within submission,
    // but we need the full span to complete before starting the next)
    await device.queue.onSubmittedWorkDone();
  }

  // Read back alpha chart
  const alphaF32 = await readBuffer(device, alphaBuffer, chartSize * 4);
  const alpha = new Float64Array(chartSize);
  for (let i = 0; i < chartSize; i++) alpha[i] = alphaF32[i];

  // Clean up
  alphaBuffer.destroy();
  singleBuffer.destroy();
  pairedBuffer.destroy();
  rulesBuffer.destroy();
  paramBuffer.destroy();

  const logLikelihood = alpha[0 * S * S + 0 * S + C];  // alpha[start=0, 0, C]
  return { alpha, logLikelihood, S };
}

// ── SCFG Viterbi (GPU) ─────────────────────────────────────────────

export async function scfgViterbiGpu(
  cg: CompiledGrammar, tw: TerminalWeights
): Promise<ParseResult> {
  const device = await getDevice();
  const K = cg.K;
  const C = tw.C;
  const S = C + 1;
  const chartSize = K * S * S;

  const rules = classifyRules(cg);
  const orderedRules = [
    ...rules.terminal, ...rules.emitUnary, ...rules.emitPaired,
    ...rules.binary, ...rules.unary,
  ];
  const packedRules = packRules(cg, orderedRules);
  const kPaired = tw.paired ? tw.paired.length : 0;

  // Initialize chart with NEG_INF
  const vInit = new Float32Array(chartSize).fill(NEG_INF);

  // Epsilon rules on CPU
  for (const r of rules.epsilon) {
    const lhs = cg.ruleLhs[r];
    const logW = cg.ruleLogWeights[r];
    for (let i = 0; i <= C; i++) {
      const idx = lhs * S * S + i * S + i;
      if (logW > vInit[idx]) vInit[idx] = logW;
    }
  }

  // Span-0 unary on CPU
  for (let pass = 0; pass < K; pass++) {
    let changed = false;
    for (const r of rules.unary) {
      const lhs = cg.ruleLhs[r];
      const rhs = cg.ruleRhs[r][0];
      const logW = cg.ruleLogWeights[r];
      for (let i = 0; i <= C; i++) {
        const childVal = vInit[rhs * S * S + i * S + i];
        if (childVal > NEG_INF + 1e30) {
          const score = logW + childVal;
          const idx = lhs * S * S + i * S + i;
          if (score > vInit[idx] + 1e-6) {
            vInit[idx] = score;
            changed = true;
          }
        }
      }
    }
    if (!changed) break;
  }

  const vBuffer = createStorageBuffer(device, vInit,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

  // Backpointer buffer
  const bpBuffer = createEmptyBuffer(device, chartSize * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // Emission weight buffers
  const singleFlat = new Float32Array(tw.single.length * C);
  for (let m = 0; m < tw.single.length; m++) {
    for (let c = 0; c < C; c++) singleFlat[m * C + c] = tw.single[m][c];
  }
  const singleBuffer = createStorageBuffer(device, singleFlat);

  const pairedSize = kPaired > 0 ? kPaired * C * C : 1;
  const pairedFlat = new Float32Array(pairedSize);
  if (tw.paired) {
    for (let m = 0; m < kPaired; m++) {
      for (let c = 0; c < C * C; c++) pairedFlat[m * C * C + c] = tw.paired[m][c];
    }
  }
  const pairedBuffer = createStorageBuffer(device, pairedFlat);

  const rulesBuffer = createStorageBuffer(device, packedRules);
  const paramBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Compile shader with 6 bindings
  const module = createShaderModule(device, SCFG_VITERBI_SHADER);

  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  const emitPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: 'emit_step' },
  });
  const binaryPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: 'binary_step' },
  });
  const unaryPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: 'unary_propagate' },
  });

  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: vBuffer } },
      { binding: 1, resource: { buffer: singleBuffer } },
      { binding: 2, resource: { buffer: pairedBuffer } },
      { binding: 3, resource: { buffer: rulesBuffer } },
      { binding: 4, resource: { buffer: bpBuffer } },
      { binding: 5, resource: { buffer: paramBuffer } },
    ],
  });

  // Span wavefront
  for (let span = 1; span <= C; span++) {
    const nPositions = C - span + 1;
    const params = new Uint32Array([
      K, C, S, span,
      rules.terminal.length, rules.emitUnary.length,
      rules.emitPaired.length, rules.binary.length,
      rules.unary.length, kPaired, 0, 0,
    ]);
    device.queue.writeBuffer(paramBuffer, 0, params);

    const encoder = device.createCommandEncoder();

    const pass1 = encoder.beginComputePass();
    pass1.setPipeline(emitPipeline);
    pass1.setBindGroup(0, bindGroup);
    pass1.dispatchWorkgroups(nPositions);
    pass1.end();

    if (rules.binary.length > 0) {
      const pass2 = encoder.beginComputePass();
      pass2.setPipeline(binaryPipeline);
      pass2.setBindGroup(0, bindGroup);
      pass2.dispatchWorkgroups(nPositions, rules.binary.length);
      pass2.end();
    }

    if (rules.unary.length > 0) {
      const pass3 = encoder.beginComputePass();
      pass3.setPipeline(unaryPipeline);
      pass3.setBindGroup(0, bindGroup);
      pass3.dispatchWorkgroups(nPositions);
      pass3.end();
    }

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  // Read back chart and backpointers
  const vF32 = await readBuffer(device, vBuffer, chartSize * 4);
  const bpReadback = device.createBuffer({
    size: chartSize * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(bpBuffer, 0, bpReadback, 0, chartSize * 4);
  device.queue.submit([enc.finish()]);
  await bpReadback.mapAsync(GPUMapMode.READ);
  const bpData = new Uint32Array(bpReadback.getMappedRange().slice(0));
  bpReadback.unmap();
  bpReadback.destroy();

  const logProb = vF32[0 * S * S + 0 * S + C];

  // Traceback on CPU using backpointers
  const labels = new Int32Array(C);
  if (logProb > NEG_INF + 1e30) {
    tracebackGpu(labels, vF32, bpData, orderedRules, cg, S, C, 0, 0, C);
  }

  // Clean up
  vBuffer.destroy();
  bpBuffer.destroy();
  singleBuffer.destroy();
  pairedBuffer.destroy();
  rulesBuffer.destroy();
  paramBuffer.destroy();

  return { labels, logProb };
}

// ── CPU traceback from GPU backpointers ─────────────────────────────

function tracebackGpu(
  labels: Int32Array,
  v: Float32Array,
  bp: Uint32Array,
  orderedRules: number[],
  cg: CompiledGrammar,
  S: number, C: number,
  nt: number, i: number, j: number,
): void {
  if (i >= j) return;

  const idx = nt * S * S + i * S + j;
  if (v[idx] <= NEG_INF + 1e30) return;

  const packed = bp[idx];
  const packedRule = packed & 0xFFFF;
  const split = (packed >> 16) & 0xFFFF;

  // Map packed rule index back to original rule
  if (packedRule >= orderedRules.length) return;
  const r = orderedRules[packedRule];

  const nRhs = cg.ruleNRhs[r];
  const nEmit = cg.ruleNEmissions[r];
  const lhs = cg.ruleLhs[r];

  if (nRhs === 0 && nEmit > 0) {
    labels[i] = lhs;
  } else if (nRhs === 1 && nEmit === 0) {
    tracebackGpu(labels, v, bp, orderedRules, cg, S, C, cg.ruleRhs[r][0], i, j);
  } else if (nRhs === 1 && nEmit > 0) {
    const nPos = cg.ruleEmissionNPos[r][0];
    if (nPos === 2) {
      labels[i] = lhs;
      labels[j - 1] = lhs;
      tracebackGpu(labels, v, bp, orderedRules, cg, S, C, cg.ruleRhs[r][0], i + 1, j - 1);
    } else {
      labels[i] = lhs;
      tracebackGpu(labels, v, bp, orderedRules, cg, S, C, cg.ruleRhs[r][0], i + 1, j);
    }
  } else if (nRhs >= 2) {
    tracebackGpu(labels, v, bp, orderedRules, cg, S, C, cg.ruleRhs[r][0], i, split);
    tracebackGpu(labels, v, bp, orderedRules, cg, S, C, cg.ruleRhs[r][1], split, j);
  }
}

// ── f32 logaddexp (for CPU epsilon/unary init) ──────────────────────

function logaddexpF32(a: number, b: number): number {
  if (a <= NEG_INF + 1e30) return b;
  if (b <= NEG_INF + 1e30) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

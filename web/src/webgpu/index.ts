/**
 * WebGPU-accelerated algorithms.
 */

export { getDevice, releaseDevice } from './device.js';
export {
  hmmForwardGpu,
  hmmBackwardGpu,
  hmmViterbiGpu,
  hmmPosteriorsGpu,
} from './hmm-gpu.js';

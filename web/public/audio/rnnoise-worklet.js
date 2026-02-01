/*
Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
*/

import createRNNWasmModuleSync from '/rnnoise/rnnoise-sync.js';

/* global AudioWorkletProcessor, registerProcessor */
class RNNoiseProcessor extends AudioWorkletProcessor {
    constructor(options = {}) {
        super();
        const { processorOptions = {} } = options;
        this.noiseGateThreshold = processorOptions.noiseGateThreshold ?? 0.012;
        this.rnnoise = null;

        // Round robin buffer for 128 (Web Audio) -> 480 (RNNoise)
        this.inputBuffer = new Float32Array(480);
        this.outputBuffer = new Float32Array(480);
        this.bufferPtr = 0;

        this.init();
    }

    init() {
        try {
            const mod = createRNNWasmModuleSync();

            this.mod = mod;
            // 0 (null) creates default model
            this.state = mod._rnnoise_create(0);

            this.frameSize = 480;
            this.heapInputPtr = mod._malloc(this.frameSize * 4);
            this.heapOutputPtr = mod._malloc(this.frameSize * 4);

            this.rnnoise = {
                processFrame: (input, output) => {
                    // Copy input to WASM heap
                    this.mod.HEAPF32.set(input, this.heapInputPtr >> 2);

                    // Process
                    this.mod._rnnoise_process_frame(this.state, this.heapOutputPtr, this.heapInputPtr);

                    // Copy output from WASM heap
                    output.set(this.mod.HEAPF32.subarray(
                        this.heapOutputPtr >> 2,
                        (this.heapOutputPtr >> 2) + this.frameSize
                    ));
                }
            };
            console.log("RNNoise initialized");
        } catch (e) {
            console.error("Failed to load RNNoise WASM", e);
        }
    }

    // Enhanced Noise Gate with smooth transition (smoothing for Whisper)
    applyNoiseGate(inputChannel, outputChannel) {
        let sum = 0;
        for (let i = 0; i < inputChannel.length; i++) sum += inputChannel[i] ** 2;
        const rms = Math.sqrt(sum / inputChannel.length);

        // Smooth gain instead of hard switch
        this.currentGain = (rms < this.noiseGateThreshold) ? 0.2 : 1.0;

        for (let i = 0; i < inputChannel.length; i++) {
            outputChannel[i] = inputChannel[i] * this.currentGain;
        }
    }

    process(inputs, outputs) {
        const input = inputs[0]?.[0];
        const output = outputs[0]?.[0];

        if (!input || !output) return true;

        if (this.rnnoise) {
            // Accumulate samples up to 480
            for (let i = 0; i < input.length; i++) {
                this.inputBuffer[this.bufferPtr] = input[i];
                output[i] = this.outputBuffer[this.bufferPtr]; // Return previous processed block
                this.bufferPtr++;

                if (this.bufferPtr === 480) {
                    // As soon as we have 480 samples, process them
                    this.rnnoise.processFrame(this.inputBuffer, this.outputBuffer);
                    this.bufferPtr = 0;
                }
            }
        } else {
            // Bypass processing if RNNoise instance is missing
            output.set(input);
        }

        return true;
    }
}

registerProcessor('rnnoise-processor', RNNoiseProcessor);
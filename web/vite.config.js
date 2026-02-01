import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
    plugins: [
        basicSsl(),
        viteStaticCopy({
            targets: [
                // VAD browser bundle + worklet + models
                { src: "node_modules/@ricky0123/vad-web/dist/bundle.min.js", dest: "vad" },
                { src: "node_modules/@ricky0123/vad-web/dist/vad.worklet.bundle.min.js", dest: "vad" },
                { src: "node_modules/@ricky0123/vad-web/dist/silero_vad_v5.onnx", dest: "vad" },
                { src: "node_modules/@ricky0123/vad-web/dist/silero_vad_legacy.onnx", dest: "vad" },

                // ONNX Runtime Web (WASM) assets
                // easiest: copy the whole dist folder (safe, slightly bigger)
                { src: "node_modules/onnxruntime-web/dist/*", dest: "ort" },

                // RNNoise assets
                { src: "node_modules/@jitsi/rnnoise-wasm/dist/rnnoise-sync.js", dest: "rnnoise" },
            ]
        })
    ],
    server: {
        headers: {
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
        },
        proxy: {
            '/auth': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
            },
            '/api': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
            },
            '/ws': {
                target: 'ws://127.0.0.1:8000',
                ws: true,
                changeOrigin: true,
            },
        },
        host: '0.0.0.0',
        allowedHosts: ['localhost', '127.0.0.1', '0.0.0.0', 'rogflow.lan'],
        port: 5173,
        strictPort: true,
    },
    optimizeDeps: {
        exclude: ['onnxruntime-web']
    },
});

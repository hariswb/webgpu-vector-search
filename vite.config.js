import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { playwright } from "@vitest/browser-playwright";

export default defineConfig({
  server: {
    port: 5173,
  },
  build: {
    target: "esnext", // Important for WebGPU
    sourcemap: true,
  },
  plugins: [react(), tailwindcss()],
  test: {
    include: [
      'src/test/pipelineLatest.test.ts'
    ],
    browser: {
      provider: playwright({
        launchOptions: {
          args: [
            "--enable-webgpu-developer-features",
            "--enable-unsafe-webgpu",
          ],
        },
      }),
      enabled: true,
      headless: true,
      instances: [
        {
          browser: "chromium",
        }
      ],
    },
  },
});

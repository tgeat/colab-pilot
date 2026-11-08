import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri 期望固定端口，端口被占用时直接失败
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: { ignored: ["**/src-tauri/**", "**/legacy/**"] },
  },
  build: {
    target: "es2021",
    outDir: "dist",
  },
});

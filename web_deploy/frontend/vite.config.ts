import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands where the gateway serves it from. In development, open http://localhost:5173 with the gateway
// running locally (GATEWAY_HTTPS_ONLY=0, see web_deploy/README.md): the proxy keeps the Host header, so the
// gateway's same-origin check sees the address the browser used.
const gateway = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "../gateway/static", emptyOutDir: true },
  server: {
    proxy: {
      "/auth": { target: gateway, changeOrigin: false },
      "/box": { target: gateway, changeOrigin: false },
      "/api": { target: gateway, changeOrigin: false },
      "/admin/api": { target: gateway, changeOrigin: false },
      "/firm/api": { target: gateway, changeOrigin: false },
      "/public": { target: gateway, changeOrigin: false },
    },
  },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/test-setup.ts"] },
});

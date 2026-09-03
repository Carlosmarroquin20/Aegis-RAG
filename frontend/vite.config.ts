import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API base URL is read from VITE_API_BASE_URL at build time (see .env.example).
// In local dev, requests to /api are proxied to the backend to avoid CORS friction.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": "http://localhost:8000",
      "/ready": "http://localhost:8000",
    },
  },
});

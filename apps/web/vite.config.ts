import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 개발 중에는 /api 요청을 ragkit-api(기본 8000 포트)로 넘긴다
const apiTarget = process.env.RAGKIT_API_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  server: { proxy: { '/api': { target: apiTarget, changeOrigin: true } } },
  build: { chunkSizeWarningLimit: 800 },
})

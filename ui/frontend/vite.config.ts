import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev flow: `python -m ui.server` on 8123 plus `npm run dev` here. The proxy
// keeps the frontend on a same-origin /api in both dev and built modes.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8123' },
  },
})

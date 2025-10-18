import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5173', // 后端 API 服务的地址
        changeOrigin: true, // 是否修改请求源
        pathRewrite: { '^/api': '' }, // 重写路径（修正拼写）
      },
    },
  },
})







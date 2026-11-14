import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import {fileURLToPath} from 'node:url';

const apps = {live: 1421, annotator: 1420};
const path = p => fileURLToPath(new URL(p, import.meta.url));
export default defineConfig(({mode}) => {
  const app = mode in apps ? mode : 'live';
  return {
    plugins: [react(), tailwindcss()],
    root: path(`ui/${app}`),
    publicDir: path('public'),
    resolve: {alias: {'@': path('ui/shared')}},
    clearScreen: false,
    // One dependency cache per app: both dev servers can run at once without one
    // re-optimising dependencies under the other.
    cacheDir: path(`node_modules/.vite/${app}`),
    build: {outDir: path(`dist/${app}`), emptyOutDir: true},
    server: {
      host: '127.0.0.1', port: apps[app], strictPort: true,
      watch: {ignored: ['**/target/**', '**/work/**', '**/data/**', '**/models/**', '**/apps/**', '**/crates/**']},
    },
  };
});

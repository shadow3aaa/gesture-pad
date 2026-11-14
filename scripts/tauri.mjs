import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const [app, ...args] = process.argv.slice(2);
if (!['live', 'annotator'].includes(app)) {
  console.error('usage: bun scripts/tauri.mjs <live|annotator> <tauri args...>');
  process.exit(1);
}
const root = fileURLToPath(new URL('..', import.meta.url));
const child = spawn('bun', ['x', 'tauri', ...args], {
  cwd: root,
  stdio: 'inherit',
  shell: true,
  env: {...process.env, TAURI_APP_PATH: `apps/${app}`, TAURI_FRONTEND_PATH: '.'},
});
child.on('exit', code => process.exit(code ?? 1));

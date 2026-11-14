import {useEffect, useRef, useState} from 'react';
import {invoke, isTauri} from '@tauri-apps/api/core';
import {emitTo} from '@tauri-apps/api/event';
import {CameraIcon, HandIcon, MoonIcon, MousePointer2Icon, SquareIcon, SunIcon, VideoOffIcon} from 'lucide-react';
import {toast} from 'sonner';
import {AsyncButton, errorText} from '@/components/async-button';
import {Badge} from '@/components/ui/badge';
import {Button} from '@/components/ui/button';
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from '@/components/ui/card';
import {Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle} from '@/components/ui/empty';
import {Field, FieldDescription, FieldLabel} from '@/components/ui/field';
import {Kbd} from '@/components/ui/kbd';
import {Spinner} from '@/components/ui/spinner';
import {DEFAULT_SENSITIVITY, HandControl, PINCH, drawHand, none} from '@/hand';
import {LABELS, META} from '@/labels';
import {useTheme} from '@/theme';
import {cameraError, detector, detect, openCamera, paint} from '@/vision';

interface Status {armed: boolean; epoch: number}
const phases = {none: '画面中没有手', idle: '空闲 · 不操作', point: '指向', press: '按下 · 可拖动', scroll: '滚动'};
type Intent = {state: 'rules'} | {state: 'loading'} | {state: 'model'; path: string; probs: number[] | null};

function loadSensitivity() {
  const saved = Number(localStorage.getItem('hand-sensitivity') ?? DEFAULT_SENSITIVITY);
  return Number.isFinite(saved) && saved >= 1 && saved <= 3 ? saved : DEFAULT_SENSITIVITY;
}

/** Live thumb–index gap with the press and release thresholds, to see why a pinch did or did not click. */
function PinchMeter({value, down}: {value: number | null; down: boolean}) {
  const full = 0.8, at = (v: number) => `${Math.min(1, v / full) * 100}%`;
  return (
    <div className="mt-4 flex flex-col gap-1.5">
      <div className="flex justify-between text-xs text-muted-foreground"><span>捏合距离</span><span className="font-mono" data-testid="pinch">{value === null ? '—' : value.toFixed(2)}</span></div>
      <div className="relative h-2 rounded-full bg-muted">
        {value !== null && <div className="h-full rounded-full transition-[width] duration-75" style={{width: at(value), background: down ? '#f59e0b' : '#22c55e'}} />}
        <span className="absolute -inset-y-1 w-0.5 bg-foreground" style={{left: at(PINCH.press)}} title="低于此值按下" />
        <span className="absolute -inset-y-1 w-0.5 bg-muted-foreground" style={{left: at(PINCH.release)}} title="高于此值松开" />
      </div>
      <p className="text-xs text-muted-foreground">低于深色刻度按下，高于浅色刻度松开。</p>
    </div>
  );
}

/** Model output per class, so a wrong decision shows which class won and by how much. */
function IntentBars({probs}: {probs: number[] | null}) {
  return (
    <div className="flex flex-col gap-1.5" data-testid="intent">
      {LABELS.map((label, i) => (
        <div key={label} className="grid grid-cols-[2.5rem_1fr_2.5rem] items-center gap-2 text-xs">
          <span>{META[label].name}</span>
          <div className="h-1.5 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full" style={{width: `${(probs?.[i] ?? 0) * 100}%`, background: META[label].color}} /></div>
          <span className="text-right font-mono text-muted-foreground">{probs ? (probs[i] * 100).toFixed(0) : '—'}</span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const {theme, toggleTheme} = useTheme();
  const desktop = isTauri();
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const screen = useRef<HTMLCanvasElement>(null);
  const control = useRef(new HandControl());
  const status = useRef<Status>({armed: false, epoch: 0});
  const requested = useRef(false);
  const [armed, setArmed] = useState(false);
  const [running, setRunning] = useState(false);
  const [camera, setCamera] = useState<'off' | 'starting' | 'ready'>('off');
  const [error, setError] = useState<string | null>(null);
  const [sample, setSample] = useState(none);
  const [fps, setFps] = useState(0);
  const [sensitivity, setSensitivity] = useState(loadSensitivity);
  const gain = useRef(sensitivity);
  gain.current = sensitivity;
  const [intent, setIntent] = useState<Intent>({state: 'rules'});
  const useModel = useRef(false);
  useModel.current = intent.state === 'model';
  const probs = useRef<number[] | null>(null);

  const loadModel = async (path: string) => {
    setIntent({state: 'loading'});
    try {
      await invoke('load_model', {path});
    } catch (e) {
      setIntent({state: 'rules'});
      throw e;
    }
    probs.current = null;
    localStorage.setItem('model-path', path);
    setIntent({state: 'model', path, probs: null});
  };
  const chooseModel = async () => {
    const {open} = await import('@tauri-apps/plugin-dialog');
    const path = await open({directory: true, title: '选择模型目录'});
    if (typeof path === 'string') await loadModel(path);
  };
  const useRules = async () => {
    await invoke('unload_model');
    probs.current = null;
    localStorage.setItem('model-path', ''); // Remembered: no model, not even the default one.
    setIntent({state: 'rules'});
  };
  useEffect(() => {
    if (!desktop) return;
    // The model picked last time; if none was ever picked, the one trained into data/model.
    const saved = localStorage.getItem('model-path');
    if (saved === '') return;
    void (async () => {
      const path = saved ?? await invoke<string | null>('default_model_dir');
      if (path) await loadModel(path);
    })().catch(e => toast.error(`没能加载模型：${errorText(e)}`));
  }, []);

  const acceptStatus = (next: Status) => {
    if (next.epoch < status.current.epoch) return;
    const accepted = {...next, armed: next.armed && requested.current};
    status.current = accepted;
    setArmed(accepted.armed);
  };
  const stopControl = () => {
    requested.current = false;
    status.current = {...status.current, armed: false};
    setArmed(false);
    if (desktop) void invoke<Status>('set_armed', {armed: false}).then(acceptStatus).catch(e => toast.error(errorText(e)));
  };

  useEffect(() => {
    if (!running) return;
    const abort = new AbortController();
    let model: Awaited<ReturnType<typeof detector>> | null = null;
    let handle = 0, inFlight = false, classifying = false, sequence = 0, frames = 0, since = performance.now();
    const v = video.current!, c = canvas.current!;
    const fail = (e: unknown) => {
      if (abort.signal.aborted) return;
      stopControl();
      setError(cameraError(e));
      setRunning(false);
      setCamera('off');
    };
    const loop = () => {
      if (abort.signal.aborted || !model) return;
      const now = performance.now();
      try {
        const frame = detect(model, v, now);
        if (desktop && useModel.current && !classifying) {
          // The GCN runs in the Rust backend; its newest probabilities drive the next update.
          classifying = true;
          invoke<number[] | null>('classify', {frame})
            .then(p => { probs.current = p; })
            .catch(e => { probs.current = null; setIntent({state: 'rules'}); toast.error(`模型推理失败，已改用规则：${errorText(e)}`); })
            .finally(() => { classifying = false; });
        }
        const next = control.current.update(frame, gain.current, useModel.current && frame.image ? probs.current : null);
        setSample(next);
        if (c.width !== v.videoWidth || c.height !== v.videoHeight) { c.width = v.videoWidth; c.height = v.videoHeight; }
        paint(c, frame, next);
        const s = screen.current;
        if (s) drawHand(s.getContext('2d')!, s.width, s.height, next);
        frames++;
        if (now - since >= 1000) { setFps(Math.round(frames * 1000 / (now - since))); frames = 0; since = now; }
        if (desktop && status.current.armed) {
          void emitTo('overlay', 'hand', next).catch(() => {});
          if (!inFlight) {
            inFlight = true;
            const {active, down, x, y, scroll_id, scroll_y} = next;
            invoke<Status>('update_pointer', {sample: {active, down, x, y, scroll_id, scroll_y, epoch: status.current.epoch, sequence: ++sequence}})
              .then(st => { if (!abort.signal.aborted) acceptStatus(st); })
              .catch(fail).finally(() => { inFlight = false; });
          }
        }
        handle = v.requestVideoFrameCallback(loop);
      } catch (e) { fail(e); }
    };
    void (async () => {
      try {
        const loaded = await detector();
        if (abort.signal.aborted) { loaded.close(); return; }
        model = loaded;
        const stream = await openCamera(v, abort.signal);
        if (abort.signal.aborted) return;
        stream.getVideoTracks()[0].addEventListener('ended', () => fail(new Error('摄像头已断开')), {once: true});
        setCamera('ready');
        handle = v.requestVideoFrameCallback(loop);
      } catch (e) { fail(e); }
    })();
    // Judge the camera by whether its picture advances, not by how fast detection runs.
    // Pointer safety on slow frames is the backend's 500ms no-data watchdog.
    let videoTime = -1, progressed = performance.now();
    const watchdog = setInterval(() => {
      if (v.readyState < 2) return;
      if (v.currentTime !== videoTime) { videoTime = v.currentTime; progressed = performance.now(); }
      else if (performance.now() - progressed > 3000) fail(new Error('摄像头画面中断，已停止控制'));
    }, 500);
    return () => {
      abort.abort();
      clearInterval(watchdog);
      if (handle) v.cancelVideoFrameCallback(handle);
      v.pause();
      v.srcObject = null;
      model?.close();
      control.current.reset();
      if (desktop) void invoke('set_armed', {armed: false}).catch(() => {});
    };
  }, [running]);

  useEffect(() => {
    const visibility = () => { if (document.hidden) stopControl(); };
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') stopControl(); };
    document.addEventListener('visibilitychange', visibility);
    window.addEventListener('keydown', key);
    let disposed = false;
    const poll = desktop ? setInterval(() => {
      void invoke<Status>('pointer_status').then(next => { if (!disposed) acceptStatus(next); }).catch(() => {});
    }, 250) : undefined;
    return () => {
      disposed = true;
      clearInterval(poll);
      document.removeEventListener('visibilitychange', visibility);
      window.removeEventListener('keydown', key);
    };
  }, []);

  const startCamera = () => { setError(null); setCamera('starting'); setFps(0); setRunning(true); };
  const closeCamera = () => { stopControl(); setRunning(false); setCamera('off'); setFps(0); };
  const toggleControl = async () => {
    if (status.current.armed) { stopControl(); return; }
    control.current.reset();
    requested.current = true;
    try {
      const next = desktop ? await invoke<Status>('set_armed', {armed: true}) : {armed: true, epoch: status.current.epoch + 1};
      acceptStatus(next);
      if (!requested.current && desktop) await invoke('set_armed', {armed: false});
    } catch (e) { requested.current = false; throw e; }
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-14 shrink-0 items-center justify-between border-b px-5">
        <div className="flex items-center gap-3"><HandIcon className="size-5" /><h1 className="font-semibold">Gesture Pad</h1></div>
        <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="切换主题">{theme === 'light' ? <MoonIcon /> : <SunIcon />}</Button>
      </header>
      <main className="grid min-h-0 flex-1 gap-5 overflow-auto p-5 lg:grid-cols-[minmax(0,1fr)_20rem] lg:overflow-hidden">
        <div className="flex min-w-0 flex-col gap-4 lg:min-h-0">
          <section className="relative min-h-64 flex-1 overflow-hidden rounded-xl border bg-muted" aria-label="摄像头预览">
            <video ref={video} autoPlay muted playsInline className="absolute inset-0 size-full -scale-x-100 object-contain" />
            <canvas ref={canvas} className="pointer-events-none absolute inset-0 size-full -scale-x-100 object-contain" />
            {camera !== 'ready' && (
              <Empty className="absolute inset-0 bg-muted">
                <EmptyHeader>
                  <EmptyMedia variant="icon">{camera === 'starting' ? <Spinner /> : error ? <VideoOffIcon /> : <CameraIcon />}</EmptyMedia>
                  <EmptyTitle>{camera === 'starting' ? '正在启动摄像头' : error ? '摄像头不可用' : '让摄像头看到你的手'}</EmptyTitle>
                  <EmptyDescription>{error ?? (camera === 'starting' ? '正在加载手部识别模型…' : '画面只在本机处理，不录制。')}</EmptyDescription>
                </EmptyHeader>
                <EmptyContent><Button onClick={camera === 'starting' ? closeCamera : startCamera}>{camera === 'starting' ? '取消启动' : error ? '重试摄像头' : '打开摄像头'}</Button></EmptyContent>
              </Empty>
            )}
            {camera === 'ready' && <Badge variant="secondary" className="absolute top-3 left-3">镜像预览 · {fps || '—'} fps</Badge>}
          </section>
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border p-4">
            <div>
              <p className="font-medium" role="status">{phases[sample.phase]}{!armed && ' · 仅预览'}</p>
              <p className="mt-1 text-xs text-muted-foreground">{desktop ? armed ? '正在控制鼠标' : '没有在控制鼠标' : '浏览器里只能预览，不会动鼠标'}</p>
            </div>
            <div className="flex gap-2">
              {running && <Button variant="outline" onClick={closeCamera}>关闭摄像头</Button>}
              <AsyncButton variant={armed ? 'destructive' : 'default'} disabled={!armed && camera !== 'ready'} onClick={toggleControl}>
                {armed ? <SquareIcon data-icon="inline-start" /> : <MousePointer2Icon data-icon="inline-start" />}{armed ? '停止控制' : desktop ? '启用控制' : '启用预览'}
              </AsyncButton>
            </div>
          </div>
        </div>
        <aside className="flex flex-col gap-4 lg:min-h-0 lg:overflow-y-auto lg:pr-1">
          <Card className="shrink-0"><CardHeader><CardTitle>屏幕上的手</CardTitle><CardDescription>启用控制后，手会这样画在屏幕最上层。</CardDescription></CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="relative aspect-video overflow-hidden rounded-lg border bg-neutral-800" aria-label="屏幕预览">
                <canvas ref={screen} width={640} height={360} className="size-full" data-testid="screen-preview" />
              </div>
              <Field><FieldLabel htmlFor="sensitivity">灵敏度 <span className="ml-auto font-mono">{sensitivity.toFixed(1)}×</span></FieldLabel>
                <input id="sensitivity" type="range" min="1" max="3" step="0.1" value={sensitivity} onChange={e => { const value = Number(e.target.value); setSensitivity(value); localStorage.setItem('hand-sensitivity', String(value)); }} />
                <FieldDescription>画面中间 1/{sensitivity.toFixed(1)} 对应整个屏幕。调高了手动得少，但去屏幕边缘时手容易出画。</FieldDescription>
              </Field>
            </CardContent>
          </Card>
          <Card className="shrink-0"><CardHeader><CardTitle>手势</CardTitle><CardDescription>圆圈是作用点，点击和拖动都落在这里。</CardDescription></CardHeader>
            <CardContent><ul className="flex flex-col gap-3 text-sm text-muted-foreground">
              <li><strong className="text-foreground">指向</strong>：只伸食指，拇指随意。</li>
              <li><strong className="text-foreground">点击 / 拖动</strong>：拇指捏到食指上，捏住移动就是拖动。</li>
              <li><strong className="text-foreground">滚动</strong>：伸出食指和中指上下划，内容跟着手走。</li>
              <li><strong className="text-foreground">空闲</strong>：张开手掌、握拳或者放松，光标不动。</li>
            </ul>
              <PinchMeter value={sample.pinch} down={sample.down} />
            </CardContent>
          </Card>
          <Card className="shrink-0"><CardHeader><CardTitle>识别方式</CardTitle>
            <CardDescription>{intent.state === 'model' ? `模型：${intent.path.split(/[\\/]/).filter(Boolean).pop()}` : intent.state === 'loading' ? '正在加载模型…' : '规则'}</CardDescription></CardHeader>
            <CardContent className="flex flex-col gap-3">
              {intent.state === 'model' && <IntentBars probs={probs.current} />}
              <div className="flex gap-2">
                <AsyncButton size="sm" variant="outline" disabled={!desktop || intent.state === 'loading'} onClick={chooseModel}>{intent.state === 'model' ? '更换模型' : '加载模型'}</AsyncButton>
                {intent.state === 'model' && <AsyncButton size="sm" variant="ghost" onClick={useRules}>改用规则</AsyncButton>}
              </div>
              {!desktop && <p className="text-xs text-muted-foreground">模型只能在桌面版里用。</p>}
            </CardContent>
          </Card>
          <p className="px-1 text-xs leading-relaxed text-muted-foreground"><Kbd>Esc</Kbd> 停止。桌面版里不管焦点在哪，按 <Kbd>Ctrl+Shift+Space</Kbd> 都能停。</p>
        </aside>
      </main>
    </div>
  );
}

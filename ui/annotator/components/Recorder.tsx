import {useEffect, useRef, useState} from 'react';
import {CircleIcon, SquareIcon, VideoOffIcon} from 'lucide-react';
import {toast} from 'sonner';
import {errorText} from '@/components/async-button';
import {Badge} from '@/components/ui/badge';
import {Button} from '@/components/ui/button';
import {Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle} from '@/components/ui/empty';
import {Field, FieldLabel} from '@/components/ui/field';
import {Progress} from '@/components/ui/progress';
import {Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue} from '@/components/ui/select';
import {Spinner} from '@/components/ui/spinner';
import {HandControl, none, type HandFrame} from '@/hand';
import {LABELS, META, type Label} from '@/labels';
import {cameraError, detector, detect, openCamera, paint} from '@/vision';
import {compact, newId, type Session} from '../lib/api';
import {autoLabel} from '../lib/autolabel';

const COUNTDOWN_MS = 3000;

/**
 * Prompted capture: pick what to perform, follow the countdown, act for the duration.
 * Landmarks are recorded (never video) and labelled by rules + prompt for review.
 */
export function Recorder({onRecorded}: {onRecorded(session: Session): unknown}) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [camera, setCamera] = useState<'starting' | 'ready' | {error: string}>('starting');
  const [attempt, setAttempt] = useState(0);
  const [prompt, setPrompt] = useState<Label>(() => {
    const saved = localStorage.getItem('rec-prompt');
    return LABELS.find(l => l === saved) ?? 'point'; // A removed label (old left/right swipe) falls back.
  });
  const [seconds, setSeconds] = useState(() => Number(localStorage.getItem('rec-seconds') ?? 20));
  const [take, setTake] = useState<{phase: 'countdown' | 'recording'; elapsed: number} | null>(null);
  const recording = useRef<{start: number; frames: HandFrame[]; prompt: Label; seconds: number} | null>(null);
  const done = useRef(onRecorded);
  done.current = onRecorded;

  useEffect(() => {
    const abort = new AbortController();
    let model: Awaited<ReturnType<typeof detector>> | null = null, handle = 0;
    const v = video.current!, c = canvas.current!;
    const preview = new HandControl();
    const loop = () => {
      if (abort.signal.aborted || !model) return;
      const now = performance.now();
      try {
        const frame = detect(model, v, now);
        if (c.width !== v.videoWidth || c.height !== v.videoHeight) { c.width = v.videoWidth; c.height = v.videoHeight; }
        paint(c, frame, frame.image ? preview.update(frame, 2) : none);
        const r = recording.current;
        if (r) {
          const elapsed = now - r.start;
          if (elapsed >= 0) r.frames.push(compact({...frame, t_ms: elapsed}));
          setTake({phase: elapsed < 0 ? 'countdown' : 'recording', elapsed});
          if (elapsed >= r.seconds * 1000) finish();
        }
        handle = v.requestVideoFrameCallback(loop);
      } catch (e) { setCamera({error: cameraError(e)}); }
    };
    void (async () => {
      try {
        const loaded = await detector();
        if (abort.signal.aborted) { loaded.close(); return; }
        model = loaded;
        await openCamera(v, abort.signal);
        if (abort.signal.aborted) return;
        setCamera('ready');
        handle = v.requestVideoFrameCallback(loop);
      } catch (e) { if (!abort.signal.aborted) setCamera({error: cameraError(e)}); }
    })();
    return () => {
      abort.abort();
      if (handle) v.cancelVideoFrameCallback(handle);
      v.pause();
      v.srcObject = null;
      model?.close();
      recording.current = null;
    };
  }, [attempt]);

  const start = () => {
    localStorage.setItem('rec-prompt', prompt);
    localStorage.setItem('rec-seconds', String(seconds));
    recording.current = {start: performance.now() + COUNTDOWN_MS, frames: [], prompt, seconds};
    setTake({phase: 'countdown', elapsed: -COUNTDOWN_MS});
  };
  const cancel = () => { recording.current = null; setTake(null); };
  function finish() {
    const r = recording.current;
    recording.current = null;
    setTake(null);
    if (!r) return;
    const withHand = r.frames.filter(f => f.image).length;
    if (withHand < 30) { toast.error('几乎没有检测到手，本次录制已丢弃'); return; }
    const session: Session = {version: 1, id: newId(), prompt: r.prompt, reviewed: false, frames: r.frames, segments: autoLabel(r.frames, r.prompt)};
    Promise.resolve(done.current(session)).catch(e => toast.error(errorText(e)));
  }

  const busy = take !== null;
  return (
    <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <section className="relative min-h-64 overflow-hidden rounded-xl border bg-muted" aria-label="摄像头预览">
        <video ref={video} autoPlay muted playsInline className="absolute inset-0 size-full -scale-x-100 object-contain" />
        <canvas ref={canvas} className="pointer-events-none absolute inset-0 size-full -scale-x-100 object-contain" />
        {camera !== 'ready' && (
          <Empty className="absolute inset-0 bg-muted">
            <EmptyHeader>
              <EmptyMedia variant="icon">{camera === 'starting' ? <Spinner /> : <VideoOffIcon />}</EmptyMedia>
              <EmptyTitle>{camera === 'starting' ? '正在启动摄像头' : '摄像头不可用'}</EmptyTitle>
              {camera !== 'starting' && <EmptyDescription>{camera.error}</EmptyDescription>}
            </EmptyHeader>
            {camera !== 'starting' && <Button variant="outline" onClick={() => { setCamera('starting'); setAttempt(n => n + 1); }}>重试</Button>}
          </Empty>
        )}
        {take && (
          <div className="absolute inset-x-0 top-0 flex flex-col items-center gap-2 bg-background/80 p-4 backdrop-blur" data-testid="take">
            {take.phase === 'countdown'
              ? <p className="text-2xl font-semibold">{Math.ceil(-take.elapsed / 1000)}…</p>
              : <Badge variant="destructive"><CircleIcon data-icon="inline-start" className="animate-pulse fill-current" />录制中 {(take.elapsed / 1000).toFixed(0)} / {seconds}s</Badge>}
            <p className="text-sm">{META[prompt].prompt}</p>
            <Progress value={Math.max(0, take.elapsed) / (seconds * 10)} className="w-64" />
          </div>
        )}
      </section>
      <aside className="flex flex-col gap-4 rounded-xl border p-4">
        <div>
          <h2 className="font-medium">采集</h2>
          <p className="mt-1 text-sm text-muted-foreground">选个动作，倒计时结束后照提示做。只存手的关键点，不录视频。</p>
        </div>
        <Field data-disabled={busy || undefined}>
          <FieldLabel>动作</FieldLabel>
          <Select value={prompt} onValueChange={v => setPrompt(v as Label)} disabled={busy}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent><SelectGroup>{LABELS.map(l => <SelectItem key={l} value={l}>{META[l].name}</SelectItem>)}</SelectGroup></SelectContent>
          </Select>
        </Field>
        <p className="rounded-lg bg-muted p-3 text-sm">{META[prompt].prompt}</p>
        <Field data-disabled={busy || undefined}>
          <FieldLabel>时长</FieldLabel>
          <Select value={String(seconds)} onValueChange={v => setSeconds(Number(v))} disabled={busy}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent><SelectGroup>{[10, 20, 30, 60].map(s => <SelectItem key={s} value={String(s)}>{s} 秒</SelectItem>)}</SelectGroup></SelectContent>
          </Select>
        </Field>
        <div className="mt-auto flex flex-col gap-2">
          {busy
            ? <Button variant="destructive" onClick={take?.phase === 'recording' ? finish : cancel}><SquareIcon data-icon="inline-start" />{take?.phase === 'recording' ? '提前结束并保存' : '取消'}</Button>
            : <Button onClick={start} disabled={camera !== 'ready'}><CircleIcon data-icon="inline-start" className="fill-current" />开始录制</Button>}
        </div>
      </aside>
    </div>
  );
}

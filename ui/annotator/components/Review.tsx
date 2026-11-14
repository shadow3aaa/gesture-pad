import {useEffect, useRef, useState} from 'react';
import {PauseIcon, PlayIcon, Redo2Icon, SaveIcon, EraserIcon, ScissorsIcon, Trash2Icon, Undo2Icon, UnfoldHorizontalIcon, ZoomInIcon, ZoomOutIcon} from 'lucide-react';
import {AsyncButton} from '@/components/async-button';
import {Badge} from '@/components/ui/badge';
import {Button} from '@/components/ui/button';
import {Checkbox} from '@/components/ui/checkbox';
import {Kbd} from '@/components/ui/kbd';
import {Label as FormLabel} from '@/components/ui/label';
import {EDGES, type HandFrame} from '@/hand';
import {LABELS, META, promptMeta, type Label} from '@/labels';
import type {Session} from '../lib/api';
import {MIN_SEGMENT_MS} from '../lib/autolabel';
import {absorbShort, boundaries, labelAt, moveBoundary, paint, removeBoundary, split} from '../lib/segments';
import {Timeline, type Range} from './Timeline';

interface Props {
  session: Session;
  dirty: boolean;
  onChange(session: Session): void;
  onSave(): unknown;
  onDelete(): unknown;
}

const frameIndex = (frames: readonly HandFrame[], t: number) => {
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (frames[mid].t_ms <= t) lo = mid; else hi = mid - 1; }
  return lo;
};
const clock = (ms: number) => `${Math.floor(ms / 60000)}:${(ms / 1000 % 60).toFixed(2).padStart(5, '0')}`;
const HISTORY = 200;

/** Skeleton replay plus a zoomable label timeline, with undo/redo for every edit. */
export function Review({session, dirty, onChange, onSave, onDelete}: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const duration = session.frames.at(-1)?.t_ms ?? 0;
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selection, setSelection] = useState<Range | null>(null);
  const [cut, setCut] = useState<number | null>(null);
  const [view, setView] = useState<Range>([0, duration]);
  const history = useRef<{past: Session[]; future: Session[]}>({past: [], future: []});
  const dragFrom = useRef<Session | null>(null);
  const [, rerender] = useState(0);

  useEffect(() => {
    setT(0); setPlaying(false); setSelection(null); setCut(null); setView([0, session.frames.at(-1)?.t_ms ?? 0]);
    history.current = {past: [], future: []};
  }, [session.id]);

  const edit = (next: Session) => {
    const h = history.current;
    h.past.push(session);
    if (h.past.length > HISTORY) h.past.shift();
    h.future = [];
    onChange(next);
  };
  const undo = () => {
    const h = history.current, prev = h.past.pop();
    if (!prev) return;
    h.future.push(session);
    onChange(prev);
    rerender(n => n + 1);
  };
  const redo = () => {
    const h = history.current, next = h.future.pop();
    if (!next) return;
    h.past.push(session);
    onChange(next);
    rerender(n => n + 1);
  };
  const apply = (label: Label | null) => {
    if (selection) edit({...session, reviewed: false, segments: paint(session.segments, selection[0], selection[1], label)});
  };
  const splitHere = () => {
    const next = split(session.segments, t);
    if (next.length === session.segments.length) return;
    edit({...session, reviewed: false, segments: next});
    setCut(t); setSelection(null);
  };
  const deleteCut = () => {
    if (cut === null) return;
    edit({...session, reviewed: false, segments: removeBoundary(session.segments, cut, duration)});
    setCut(null);
  };
  // A whole drag is one undo step: the history gets the state from before the drag began.
  const cutDrag = {
    start: () => { dragFrom.current = session; history.current.past.push(session); history.current.future = []; },
    move: (from: number, to: number) => {
      const base = dragFrom.current!;
      const segments = moveBoundary(base.segments, from, to, duration);
      onChange({...base, reviewed: false, segments});
      // moveBoundary keeps order, so the cut's new position is on the same left/right segment.
      const left = base.segments.findIndex(s => s.end_ms === from);
      setCut(left >= 0 ? segments[left].end_ms : segments[base.segments.findIndex(s => s.start_ms === from)].start_ms);
    },
    end: () => { dragFrom.current = null; rerender(n => n + 1); },
  };
  /** Select the next/previous segment however small it is, without having to click it. */
  const jumpSegment = (dir: -1 | 1) => {
    const list = session.segments, at = selection?.[0] ?? t;
    const next = dir > 0 ? list.find(s => s.start_ms > at + 0.5) : [...list].reverse().find(s => s.start_ms < at - 0.5);
    if (!next) return;
    setSelection([next.start_ms, next.end_ms]); setCut(null); seek(next.start_ms);
  };
  const short = session.segments.filter(s => s.end_ms - s.start_ms < MIN_SEGMENT_MS).length;
  const cleanUp = () => {
    edit({...session, reviewed: false, segments: absorbShort(session.segments, MIN_SEGMENT_MS)});
    setSelection(null); setCut(null);
  };
  const jumpCut = (dir: -1 | 1) => {
    const cuts = boundaries(session.segments);
    const next = dir > 0 ? cuts.find(b => b > t + 1) : [...cuts].reverse().find(b => b < t - 1);
    if (next !== undefined) { seek(next); setCut(next); setSelection(null); }
  };

  // Keep the playhead visible when it moves out of a zoomed view.
  const seek = (ms: number) => {
    const at = Math.max(0, Math.min(duration, ms));
    setT(at);
    setView(([a, b]) => at < a || at > b ? [Math.max(0, Math.min(duration - (b - a), at - (b - a) / 2)), Math.max(0, Math.min(duration - (b - a), at - (b - a) / 2)) + (b - a)] : [a, b]);
  };
  const zoom = (factor: number) => setView(([a, b]) => {
    const w = Math.max(400, Math.min(duration, (b - a) * factor)), c = Math.max(a, Math.min(b, t));
    const start = Math.max(0, Math.min(duration - w, c - (c - a) * w / (b - a)));
    return [start, start + w];
  });
  const stepFrames = (n: number) => {
    const i = frameIndex(session.frames, t), j = Math.max(0, Math.min(session.frames.length - 1, i + n));
    seek(session.frames[j]?.t_ms ?? 0);
  };

  // Playback in real time; loops inside the selection if there is one.
  useEffect(() => {
    if (!playing) return;
    let raf = 0, last = performance.now();
    const tick = (now: number) => {
      setT(prev => {
        const [a, b] = selection ?? [0, duration];
        const next = prev + (now - last);
        return next >= b || next < a ? a : next;
      });
      last = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, selection, duration]);

  // Draw the hand as recorded, mirrored like the live preview, coloured by its label.
  useEffect(() => {
    const c = canvas.current!, ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    const f = session.frames.length ? session.frames[frameIndex(session.frames, t)] : null;
    if (!f?.image) return;
    const label = labelAt(session.segments, f.t_ms);
    const p = f.image.map(([u, v]) => [(1 - u) * c.width, v * c.height] as const);
    ctx.strokeStyle = ctx.fillStyle = label ? META[label].color : '#e2e8f0';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (const [a, b] of EDGES) { ctx.moveTo(...p[a]); ctx.lineTo(...p[b]); }
    ctx.stroke();
    for (const q of p) { ctx.beginPath(); ctx.arc(q[0], q[1], 4, 0, Math.PI * 2); ctx.fill(); }
  }, [t, session]);

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea, [role=combobox]')) return;
      const k = e.key.toLowerCase(), n = Number(e.key), digit = Number.isInteger(n) && n >= 1 && n <= LABELS.length ? n - 1 : -1;
      if (e.ctrlKey && k === 'z') { if (e.shiftKey) redo(); else undo(); }
      else if (e.ctrlKey && k === 'y') redo();
      else if (e.ctrlKey && k === 's') void onSave();
      else if (e.ctrlKey) return;
      else if (digit >= 0) apply(LABELS[digit]);
      else if (k === 'delete' || k === 'backspace') { if (cut !== null) deleteCut(); else apply(null); }
      else if (k === 's') splitHere();
      else if (k === '[' || k === ']') jumpCut(k === '[' ? -1 : 1);
      else if (k === 'tab') jumpSegment(e.shiftKey ? -1 : 1);
      else if (k === ' ') setPlaying(p => !p);
      else if (k === 'escape') { setSelection(null); setCut(null); }
      else if (k === 'arrowleft' || k === 'arrowright') stepFrames((k === 'arrowleft' ? -1 : 1) * (e.shiftKey ? 10 : 1));
      else if (k === 'i') setSelection(s => [t, s && s[1] > t ? s[1] : Math.min(duration, t + 500)]);
      else if (k === 'o') setSelection(s => [s && s[0] < t ? s[0] : Math.max(0, t - 500), t]);
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  });

  const current = labelAt(session.segments, t);
  const {past, future} = history.current;
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="font-medium">{promptMeta(session.prompt).name}采集</h2>
        <span className="font-mono text-xs text-muted-foreground">{session.id}</span>
        {dirty && <Badge variant="outline">未保存</Badge>}
        <div className="ml-auto flex items-center gap-2">
          <Button size="icon" variant="ghost" onClick={undo} disabled={!past.length} aria-label="撤销" title="撤销 (Ctrl+Z)"><Undo2Icon /></Button>
          <Button size="icon" variant="ghost" onClick={redo} disabled={!future.length} aria-label="重做" title="重做 (Ctrl+Shift+Z)"><Redo2Icon /></Button>
          <div className="mx-1 flex items-center gap-2">
            <Checkbox id="reviewed" checked={session.reviewed} onCheckedChange={v => edit({...session, reviewed: v === true})} />
            <FormLabel htmlFor="reviewed">已校对</FormLabel>
          </div>
          <AsyncButton size="sm" onClick={onSave} disabled={!dirty}><SaveIcon data-icon="inline-start" />保存</AsyncButton>
          <AsyncButton size="sm" variant="ghost" onClick={onDelete} aria-label="删除会话"><Trash2Icon /></AsyncButton>
        </div>
      </div>
      <section className="relative min-h-48 flex-1 overflow-hidden rounded-xl border bg-neutral-900">
        <canvas ref={canvas} width={1280} height={720} className="size-full object-contain" data-testid="replay" />
        <Badge className="absolute top-3 left-3" style={current ? {background: META[current].color} : undefined} variant={current ? 'default' : 'secondary'}>{current ? META[current].name : '未标注'}</Badge>
      </section>
      <div className="flex flex-wrap items-center gap-3">
        <Button size="icon" variant="outline" onClick={() => setPlaying(p => !p)} aria-label={playing ? '暂停' : '播放'}>{playing ? <PauseIcon /> : <PlayIcon />}</Button>
        <span className="font-mono text-sm tabular-nums">{clock(t)} / {clock(duration)}</span>
        {selection && <span className="text-sm text-muted-foreground">选中 {clock(selection[0])}–{clock(selection[1])}（{((selection[1] - selection[0]) / 1000).toFixed(2)} 秒）</span>}
        <div className="ml-auto flex items-center gap-1">
          <Button size="icon" variant="ghost" onClick={() => zoom(0.5)} aria-label="放大时间轴"><ZoomInIcon /></Button>
          <Button size="icon" variant="ghost" onClick={() => zoom(2)} aria-label="缩小时间轴"><ZoomOutIcon /></Button>
          <Button size="sm" variant="ghost" onClick={() => setView([0, duration])}>全部</Button>
        </div>
      </div>
      <Timeline duration={duration} segments={session.segments} t={t} selection={selection} cut={cut} view={view}
        onSeek={seek} onSelect={setSelection} onCut={setCut} onCutDrag={cutDrag} onView={setView} />
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="outline" onClick={splitHere}><ScissorsIcon data-icon="inline-start" />在播放处切分<Kbd>S</Kbd></Button>
        <Button size="sm" variant="outline" disabled={cut === null} onClick={deleteCut}><UnfoldHorizontalIcon data-icon="inline-start" />删除边界<Kbd>Del</Kbd></Button>
        <Button size="sm" variant="outline" disabled={!short} onClick={cleanUp} title={`短于 ${MIN_SEGMENT_MS}ms 的片段并到旁边较长的一段`}><EraserIcon data-icon="inline-start" />清理碎片{short > 0 && `（${short}）`}</Button>
        <span className="mx-1 w-px self-stretch bg-border" />
        {LABELS.map((l, i) => (
          <Button key={l} size="sm" variant="outline" disabled={!selection} onClick={() => apply(l)}>
            <span className="size-2.5 rounded-full" style={{background: META[l].color}} />{META[l].name}<Kbd>{i + 1}</Kbd>
          </Button>
        ))}
        <Button size="sm" variant="ghost" disabled={!selection} onClick={() => apply(null)}>清除<Kbd>Del</Kbd></Button>
      </div>
      <dl className="grid grid-cols-[3rem_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-xs text-muted-foreground">
        <dt className="text-foreground">选择</dt>
        <dd>单击一段，或拖动框选 · <Kbd>Tab</Kbd> 下一段 · <Kbd>I</Kbd> <Kbd>O</Kbd> 设起点和终点</dd>
        <dt className="text-foreground">剪辑</dt>
        <dd><Kbd>S</Kbd> 切一刀 · 单击边界选中，拖动移动，<Kbd>Del</Kbd> 删除（合并后用左边的标签）· <Kbd>[</Kbd> <Kbd>]</Kbd> 跳到边界</dd>
        <dt className="text-foreground">标注</dt>
        <dd><Kbd>1</Kbd>–<Kbd>{LABELS.length}</Kbd> 改标签 · <Kbd>Del</Kbd> 清除 · <Kbd>Ctrl+Z</Kbd> 撤销</dd>
        <dt className="text-foreground">查看</dt>
        <dd><Kbd>空格</Kbd> 播放 · <Kbd>←</Kbd> <Kbd>→</Kbd> 逐帧 · 滚轮缩放，<Kbd>Shift</Kbd>+滚轮平移，双击放大</dd>
      </dl>
      <p className="text-xs text-muted-foreground">划动录制里收手的那段最容易标错。勾上「已校对」的会话才会拿去训练。</p>
    </div>
  );
}

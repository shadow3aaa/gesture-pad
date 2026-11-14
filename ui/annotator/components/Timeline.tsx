import {useEffect, useRef, useState} from 'react';
import {META} from '@/labels';
import {boundaries, rangeAt, type Segment} from '../lib/segments';

export type Range = [number, number];
interface Props {
  duration: number;
  segments: readonly Segment[];
  t: number;
  selection: Range | null;
  cut: number | null;
  view: Range;
  onSeek(t: number): void;
  onSelect(range: Range | null): void;
  onCut(cut: number | null): void;
  /** Dragging a cut: `start` once, `move` with the target time, `end` when released. */
  onCutDrag: {start(cut: number): void; move(cut: number, to: number): void; end(): void};
  onView(view: Range): void;
}

const EDGE_PX = 6, SNAP_PX = 6, CLICK_PX = 4, MIN_SPAN_MS = 200;
/** Segments are drawn and hit-tested at least MIN_PX wide; below SMALL_PX a click selects the segment, not its cuts. */
const MIN_PX = 6, SMALL_PX = 18;
const TICKS = [100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000];

/**
 * Zoomable label track, edited like a video timeline. Click a segment (or gap) to select it;
 * click a cut to select it; drag a cut to move it; drag elsewhere to select a range; drag a
 * selection edge to adjust it (snaps to cuts). Wheel zooms around the pointer, Shift+wheel pans.
 */
export function Timeline({duration, segments, t, selection, cut, view, onSeek, onSelect, onCut, onCutDrag, onView}: Props) {
  const track = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<'cut' | 'edge' | null>(null);
  const [trackWidth, setTrackWidth] = useState(1);
  useEffect(() => {
    const el = track.current!, observer = new ResizeObserver(() => setTrackWidth(el.clientWidth || 1));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const [v0, v1] = view;
  const span = Math.max(1, v1 - v0);
  const x = (ms: number) => `${(ms - v0) / span * 100}%`;
  const width = () => track.current?.getBoundingClientRect().width || 1;
  /** On-screen box of a segment in px from the track's left, widened to MIN_PX around its centre. */
  const box = (s: Segment): [number, number] => {
    const left = (s.start_ms - v0) / span * trackWidth, w = (s.end_ms - s.start_ms) / span * trackWidth;
    return w >= MIN_PX ? [left, w] : [left + w / 2 - MIN_PX / 2, MIN_PX];
  };
  /** The narrowest small segment drawn under a pointer position, if any. */
  const smallAt = (clientX: number) => {
    const p = clientX - track.current!.getBoundingClientRect().left;
    return segments
      .filter(s => (s.end_ms - s.start_ms) / span * trackWidth < SMALL_PX)
      .filter(s => { const [l, w] = box(s); return p >= l && p <= l + w; })
      .sort((a, b) => (a.end_ms - a.start_ms) - (b.end_ms - b.start_ms))[0] ?? null;
  };
  const timeAt = (clientX: number) => {
    const r = track.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, v0 + (clientX - r.left) / r.width * span));
  };
  const px = (ms: number) => ms / span * width();
  const cuts = boundaries(segments);
  const nearest = (ms: number, list: readonly number[]) => {
    let best: number | null = null;
    for (const b of list) if (Math.abs(px(b - ms)) < EDGE_PX && (best === null || Math.abs(b - ms) < Math.abs(best - ms))) best = b;
    return best;
  };
  const snap = (ms: number) => {
    let best = ms;
    for (const b of cuts) if (Math.abs(px(b - ms)) < SNAP_PX && Math.abs(b - ms) <= Math.abs(best - ms)) best = b;
    return best;
  };

  // Wheel needs a non-passive listener to stop the page from scrolling.
  const latest = useRef({view, duration, onView});
  latest.current = {view, duration, onView};
  useEffect(() => {
    const el = track.current!;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const {view: [a, b], duration: d, onView: set} = latest.current;
      const w = b - a, r = el.getBoundingClientRect();
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const start = Math.max(0, Math.min(d - w, a + (e.deltaX || e.deltaY) / r.width * w));
        set([start, start + w]);
        return;
      }
      const anchor = a + (e.clientX - r.left) / r.width * w;
      const next = Math.max(MIN_SPAN_MS * 2, Math.min(d, w * Math.exp(e.deltaY * 0.002)));
      const start = Math.max(0, Math.min(d - next, anchor - (anchor - a) * next / w));
      set([start, start + next]);
    };
    el.addEventListener('wheel', wheel, {passive: false});
    return () => el.removeEventListener('wheel', wheel);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const start = timeAt(e.clientX), downX = e.clientX;
    // Dragging: a selection edge wins over a cut, then a fresh range. Clicking: a cut wins
    // (a selection edge often sits exactly on a cut, and a click there means "this cut").
    // A tiny segment under the pointer beats its own cuts and selection edges: otherwise it
    // could never be clicked. Its cuts stay reachable with [ ] or after zooming in.
    const small = smallAt(e.clientX);
    const edge = selection && !small ? [0, 1].find(i => Math.abs(px(selection[i] - start)) < EDGE_PX) : undefined;
    const near = small ? null : nearest(start, cuts);
    const grabbed = edge === undefined ? near : null;
    let moved = false;
    const move = (ev: PointerEvent) => {
      const now = timeAt(ev.clientX);
      if (!moved && Math.abs(ev.clientX - downX) < CLICK_PX) return;
      if (!moved && grabbed !== null) { onCutDrag.start(grabbed); onCut(grabbed); onSelect(null); }
      moved = true;
      if (grabbed !== null) onCutDrag.move(grabbed, now);
      else if (selection && edge !== undefined) {
        const other = selection[1 - edge], at = snap(now);
        onSelect(at < other ? [at, other] : [other, at]);
      } else {
        const [a, b] = [snap(start), snap(now)];
        onSelect(a < b ? [a, b] : [b, a]);
        onCut(null);
      }
      onSeek(now);
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      if (moved) { if (grabbed !== null) onCutDrag.end(); return; }
      if (near !== null) { onCut(near); onSelect(null); onSeek(near); return; }
      // A click selects the segment (or the unlabelled gap) under the pointer, and seeks there.
      onCut(null);
      onSelect(small ? [small.start_ms, small.end_ms] : rangeAt(segments, start, duration));
      onSeek(small ? small.start_ms : start);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };
  const onHover = (e: React.PointerEvent) => {
    const at = timeAt(e.clientX);
    if (smallAt(e.clientX)) { setHover(null); return; }
    setHover(selection && [0, 1].some(i => Math.abs(px(selection[i] - at)) < EDGE_PX) ? 'edge' : nearest(at, cuts) !== null ? 'cut' : null);
  };
  // Double-click zooms in 4× around the pointer.
  const onDoubleClick = (e: React.MouseEvent) => {
    const anchor = timeAt(e.clientX), next = Math.max(MIN_SPAN_MS * 2, span / 4);
    const start = Math.max(0, Math.min(duration - next, anchor - (anchor - v0) * next / span));
    onView([start, start + next]);
  };

  const step = TICKS.find(s => span / s <= 12) ?? 60000;
  const ticks: number[] = [];
  for (let s = Math.ceil(v0 / step) * step; s <= v1; s += step) ticks.push(s);
  return (
    <div className="flex flex-col gap-1">
      <div className="relative h-4 text-[10px] text-muted-foreground select-none">
        {ticks.map(s => <span key={s} className="absolute -translate-x-1/2 font-mono" style={{left: x(s)}}>{(s / 1000).toFixed(step < 1000 ? 2 : 0)}s</span>)}
      </div>
      <div ref={track} onPointerDown={onPointerDown} onPointerMove={onHover} onPointerLeave={() => setHover(null)} onDoubleClick={onDoubleClick}
        className="relative h-16 touch-none overflow-hidden rounded-lg border bg-muted select-none" style={{cursor: hover ? 'ew-resize' : 'crosshair'}} data-testid="timeline">
        {ticks.map(s => <div key={s} className="absolute inset-y-0 w-px bg-border" style={{left: x(s)}} />)}
        {segments.filter(s => s.end_ms > v0 && s.start_ms < v1).map(s => {
          const [left, w] = box(s);
          return <div key={`${s.start_ms}-${s.end_ms}`} data-label={s.label} className={`absolute inset-y-2 ${w <= MIN_PX ? 'z-10 rounded-sm ring-1 ring-foreground/60' : ''}`}
            style={{left, width: w, background: META[s.label].color}} title={`${META[s.label].name} ${(s.start_ms / 1000).toFixed(2)}–${(s.end_ms / 1000).toFixed(2)}s（${(s.end_ms - s.start_ms).toFixed(0)}ms）`} />;
        })}
        {cuts.filter(b => b >= v0 && b <= v1).map(b => <div key={b} data-cut={b} className="absolute inset-y-1 -ml-px w-0.5 bg-background" style={{left: x(b)}} />)}
        {cut !== null && <div className="absolute inset-y-0 -ml-[3px] w-1.5 rounded-sm bg-sky-500 ring-2 ring-background" style={{left: x(cut)}} data-testid="selected-cut" />}
        {selection && <>
          <div className="absolute inset-y-0 bg-foreground/15 ring-2 ring-foreground ring-inset" style={{left: x(selection[0]), width: `${(selection[1] - selection[0]) / span * 100}%`}} data-testid="selection" />
          {selection.map((ms, i) => <div key={i} className="absolute inset-y-0 -ml-1 w-2 rounded-sm bg-foreground" style={{left: x(ms)}} />)}
        </>}
        <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-red-500" style={{left: x(t)}} />
      </div>
      <div className="relative h-1.5 rounded-full bg-muted" aria-label="视图位置">
        <div className="absolute inset-y-0 rounded-full bg-muted-foreground/50" style={{left: `${v0 / (duration || 1) * 100}%`, width: `${span / (duration || 1) * 100}%`}} />
      </div>
    </div>
  );
}

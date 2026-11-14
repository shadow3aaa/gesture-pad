import type {Label} from '@/labels';

export interface Segment {start_ms: number; end_ms: number; label: Label}

/** Shortest segment an edit may leave behind. */
export const MIN_MS = 20;
const sorted = (s: Segment[]) => s.sort((a, b) => a.start_ms - b.start_ms);

/**
 * Overwrite [start, end) with `label` (or clear it with null). The new segment merges with
 * touching neighbours of the same label; other boundaries (e.g. from a split) are kept.
 */
export function paint(segments: readonly Segment[], start: number, end: number, label: Label | null): Segment[] {
  if (!(end > start)) return [...segments];
  const out: Segment[] = [];
  for (const s of segments) {
    if (s.end_ms <= start || s.start_ms >= end) { out.push(s); continue; }
    if (s.start_ms < start) out.push({...s, end_ms: start});
    if (s.end_ms > end) out.push({...s, start_ms: end});
  }
  if (!label) return sorted(out);
  let added: Segment = {start_ms: start, end_ms: end, label};
  const kept = out.filter(s => {
    if (s.label === label && s.end_ms === added.start_ms) { added = {...added, start_ms: s.start_ms}; return false; }
    if (s.label === label && s.start_ms === added.end_ms) { added = {...added, end_ms: s.end_ms}; return false; }
    return true;
  });
  return sorted([...kept, added]);
}

/** Every distinct segment edge, sorted: the cut points of the timeline. */
export const boundaries = (segments: readonly Segment[]) =>
  [...new Set(segments.flatMap(s => [s.start_ms, s.end_ms]))].sort((a, b) => a - b);

/** Cut the segment under `t` into two with the same label (a razor cut). */
export function split(segments: readonly Segment[], t: number): Segment[] {
  return sorted(segments.flatMap(s =>
    s.start_ms + MIN_MS <= t && t <= s.end_ms - MIN_MS ? [{...s, end_ms: t}, {...s, start_ms: t}] : [s]));
}

/**
 * Remove the cut at `b`. Two segments meeting there join (keeping the left label); a
 * segment edge next to a gap extends across the gap to the next segment or the end.
 */
export function removeBoundary(segments: readonly Segment[], b: number, duration: number): Segment[] {
  const left = segments.find(s => s.end_ms === b), right = segments.find(s => s.start_ms === b);
  if (left && right) return sorted([...segments.filter(s => s !== left && s !== right), {...left, end_ms: right.end_ms}]);
  if (left) {
    const next = Math.min(duration, ...segments.filter(s => s.start_ms >= b).map(s => s.start_ms));
    return sorted(segments.map(s => s === left ? {...s, end_ms: Math.max(next, s.end_ms)} : s));
  }
  if (right) {
    const prev = Math.max(0, ...segments.filter(s => s.end_ms <= b).map(s => s.end_ms));
    return sorted(segments.map(s => s === right ? {...s, start_ms: Math.min(prev, s.start_ms)} : s));
  }
  return [...segments];
}

/** Move the cut at `b` to `to`, adjusting the segments on both sides; never crosses a neighbour. */
export function moveBoundary(segments: readonly Segment[], b: number, to: number, duration: number): Segment[] {
  const left = segments.find(s => s.end_ms === b), right = segments.find(s => s.start_ms === b);
  if (!left && !right) return [...segments];
  const lo = left ? left.start_ms + MIN_MS : Math.max(0, ...segments.filter(s => s.end_ms <= b).map(s => s.end_ms));
  const hi = right ? right.end_ms - MIN_MS : Math.min(duration, ...segments.filter(s => s.start_ms >= b).map(s => s.start_ms));
  const at = Math.max(lo, Math.min(hi, to));
  return segments.map(s => s === left ? {...s, end_ms: at} : s === right ? {...s, start_ms: at} : s);
}

/** The labelled segment under `t`, or else the unlabelled gap around it. */
export function rangeAt(segments: readonly Segment[], t: number, duration: number): [number, number] {
  const hit = segments.find(s => s.start_ms <= t && t < s.end_ms);
  if (hit) return [hit.start_ms, hit.end_ms];
  const before = Math.max(0, ...segments.filter(s => s.end_ms <= t).map(s => s.end_ms));
  const after = Math.min(duration, ...segments.filter(s => s.start_ms > t).map(s => s.start_ms));
  return [before, after];
}

/**
 * Fold segments shorter than `minMs` (rule flicker) into a touching neighbour, the longer
 * one when there are two. Isolated short segments (gaps on both sides) are left alone.
 */
export function absorbShort(segments: readonly Segment[], minMs: number): Segment[] {
  let out = sorted(segments.map(s => ({...s})));
  for (;;) {
    const short = out
      .map((s, i) => ({s, i, left: out[i - 1]?.end_ms === s.start_ms ? out[i - 1] : null, right: out[i + 1]?.start_ms === s.end_ms ? out[i + 1] : null}))
      .filter(c => c.s.end_ms - c.s.start_ms < minMs && (c.left || c.right))
      .sort((a, b) => (a.s.end_ms - a.s.start_ms) - (b.s.end_ms - b.s.start_ms))[0];
    if (!short) return out;
    const {s, left, right} = short;
    const into = left && right ? (left.end_ms - left.start_ms >= right.end_ms - right.start_ms ? left : right) : (left ?? right)!;
    if (into === left) into.end_ms = s.end_ms; else into.start_ms = s.start_ms;
    out = out.filter(x => x !== s);
    // Neighbours that now touch with the same label become one.
    out = out.reduce<Segment[]>((acc, x) => {
      const prev = acc[acc.length - 1];
      if (prev && prev.label === x.label && prev.end_ms === x.start_ms) prev.end_ms = x.end_ms;
      else acc.push(x);
      return acc;
    }, []);
  }
}

/** Per-frame labels → segments. A frame lasts until the next one; null (no hand) leaves a gap. */
export function fromFrames(times: readonly number[], labels: readonly (Label | null)[]): Segment[] {
  const out: Segment[] = [];
  times.forEach((t, i) => {
    const label = labels[i];
    const end = times[i + 1] ?? t + 33;
    if (!label) return;
    const prev = out[out.length - 1];
    if (prev && prev.label === label && prev.end_ms === t) prev.end_ms = end;
    else out.push({start_ms: t, end_ms: end, label});
  });
  return out;
}

export const labelAt = (segments: readonly Segment[], t: number) => segments.find(s => s.start_ms <= t && t < s.end_ms)?.label ?? null;

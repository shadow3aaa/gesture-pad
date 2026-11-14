import {HandControl, type HandFrame} from '@/hand';
import {SWIPE, type Label} from '@/labels';
import {absorbShort, fromFrames, type Segment} from './segments';

/** Screen fractions per second along the prompted direction that count as a swipe stroke. */
const STROKE_SPEED = 0.35;
/** Shorter segments are rule flicker, not actions; they are folded into a neighbour. */
export const MIN_SEGMENT_MS = 150;

/**
 * Initial labels from the existing rules plus the recording prompt, for a human to correct.
 * The prompt resolves what rules cannot: an "idle" take is idle throughout, and in a swipe
 * take only fast motion in the prompted direction is the swipe — returns and pauses are idle.
 * Otherwise the rules' pose check decides: pointing or pinching is point, anything else idle.
 */
export function autoLabel(frames: readonly HandFrame[], prompt: Label): Segment[] {
  const control = new HandControl();
  const dir = SWIPE[prompt];
  const raw = frames.map((f, i) => {
    const s = control.update(f, 2);
    if (s.phase === 'none') return null;
    if (prompt === 'idle') return 'idle';
    if (dir) {
      // Velocity of the hand on screen over ±2 frames.
      const a = frames[Math.max(0, i - 2)], b = frames[Math.min(frames.length - 1, i + 2)];
      if (!a.image || !b.image || b.t_ms <= a.t_ms) return 'idle';
      const mid = (f: HandFrame) => [-(f.image![4][0] + f.image![8][0]) / 2 * 2, (f.image![4][1] + f.image![8][1]) / 2 * 2];
      const [ax, ay] = mid(a), [bx, by] = mid(b);
      const speed = ((bx - ax) * dir[0] + (by - ay) * dir[1]) / ((b.t_ms - a.t_ms) / 1000);
      return speed > STROKE_SPEED ? prompt : 'idle';
    }
    // Not an operating pose (open palm, fist, relaxed hand…): idle, whatever the prompt.
    // Pointing and pinching are both `point`: presses are decided by rules, not the model.
    return s.phase === 'idle' ? 'idle' : 'point';
  }) as (Label | null)[];
  // Majority over ±2 frames removes single-frame flicker.
  const smooth = raw.map((l, i) => {
    if (!l) return null;
    const counts = new Map<Label, number>();
    for (let j = Math.max(0, i - 2); j <= Math.min(raw.length - 1, i + 2); j++) if (raw[j]) counts.set(raw[j]!, (counts.get(raw[j]!) ?? 0) + 1);
    return [...counts].reduce((a, b) => b[1] > a[1] ? b : a)[0];
  });
  return absorbShort(fromFrames(frames.map(f => f.t_ms), smooth), MIN_SEGMENT_MS);
}

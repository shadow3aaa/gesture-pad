import {describe, expect, test} from 'bun:test';
import {LABELS, type Label} from '../ui/shared/labels';
import {HandControl, toScreen, type HandFrame, type HandSample, type Vec2, type Vec3} from '../ui/shared/hand';

/** `point` (index alone) is the default operating pose; pinch/touch/cross move its thumb. */
type Pose = 'point' | 'pinch' | 'touch' | 'cross' | 'scroll' | 'open' | 'fist';
const ASPECT = 16 / 9;
// Metric hand shape (metres): wrist at origin, fingers toward −y.
function world(pose: Pose): Vec3[] {
  const w: Vec3[] = Array.from({length: 21}, () => [0, 0, 0]);
  const finger = (mcp: number, x: number, y: number, straight: boolean) => {
    w[mcp] = [x, y, 0];
    w[mcp + 1] = [x, y - 0.03, 0];
    w[mcp + 2] = straight ? [x, y - 0.05, 0] : [x, y - 0.02, 0.02];
    w[mcp + 3] = straight ? [x, y - 0.07, 0] : [x, y, 0.02];
  };
  // Straight fingers: index, middle, ring, pinky.
  const straight = {open: [1, 1, 1, 1], scroll: [1, 1, 0, 0], fist: [0, 0, 0, 0]}[pose as string] ?? [1, 0, 0, 0];
  finger(5, 0.03, -0.08, !!straight[0]);
  finger(9, 0.01, -0.085, !!straight[1]);
  finger(13, -0.01, -0.08, !!straight[2]);
  finger(17, -0.03, -0.075, !!straight[3]);
  w[1] = [0.03, -0.02, 0]; w[2] = [0.05, -0.04, 0]; w[3] = [0.06, -0.06, 0];
  w[4] = {
    pinch: [0.034, -0.148, 0],
    // Real MediaPipe: tip landmarks are at the nail ends, so touching pads leave ~1.8 cm.
    touch: [0.03, -0.132, 0],
    // Thumb overlaps the index on screen but sits 6 cm closer to the camera.
    cross: [0.03, -0.148, 0.06],
    // Thumb wrapped over the curled index.
    fist: [0.03, -0.09, 0.02],
  }[pose as string] as Vec3 ?? [0.075, -0.09, 0];
  return w;
}
interface Options {pose?: Pose; at?: Vec2; scale?: number; noise?: () => number}
function frame(t_ms: number, {pose = 'point', at = [0.5, 0.5], scale = 2, noise}: Options = {}): HandFrame {
  const w = world(pose);
  const n = () => noise?.() ?? 0;
  // Place the hand so that the thumb–index midpoint (the action point when open) sits at `at`.
  const mid: Vec2 = [(w[4][0] + w[8][0]) / 2, (w[4][1] + w[8][1]) / 2];
  return {t_ms, aspect: ASPECT, world: w, image: w.map(([x, y]): Vec2 => [at[0] + (x - mid[0]) * scale / ASPECT + n(), at[1] + (y - mid[1]) * scale + n()])};
}
function rng(amp: number) {
  let s = 7;
  return () => { s = (s * 1103515245 + 12345) % 2147483648; return (s / 2147483648 * 2 - 1) * amp; };
}

class Sim {
  hand = new HandControl();
  now = 0;
  run(options: Options & {intent?: number[]} = {}, frames = 1, sensitivity = 2): HandSample {
    let s!: HandSample;
    for (let i = 0; i < frames; i++) s = this.hand.update(frame(this.now += 33, options), sensitivity, options.intent ?? null);
    return s;
  }
}
/** Model output (LABELS order) that is confident in one class. */
const intent = (label: Label) => LABELS.map(l => l === label ? 0.9 : 0.1 / (LABELS.length - 1));

describe('mapping', () => {
  test('mirrored, centred, scaled by sensitivity; no calibration', () => {
    expect(toScreen([0.5, 0.5], 2)).toEqual([0.5, 0.5]);
    expect(toScreen([0.75, 0.25], 2)).toEqual([0, 0]); // Image right is the user's left.
    expect(toScreen([0.6, 0.5], 1)[0]).toBeCloseTo(0.4, 9);
  });
  test('cursor follows the hand absolutely; the same hand position returns to the same place', () => {
    const s = new Sim();
    const a = s.run({at: [0.4, 0.45]}, 20);
    expect(a).toMatchObject({phase: 'point', active: true, down: false});
    expect(a.x).toBeCloseTo(0.7, 3);
    expect(a.y).toBeCloseTo(0.4, 3);
    s.run({at: [0.6, 0.6]}, 20);
    const back = s.run({at: [0.4, 0.45]}, 20);
    expect(back.x).toBeCloseTo(a.x, 3);
  });
  test('the drawn hand has a fixed size wherever and however large the hand appears', () => {
    const palm = (s: HandSample) => Math.hypot(s.skeleton![5][0] - s.skeleton![17][0], s.skeleton![5][1] - s.skeleton![17][1]);
    const near = new Sim().run({scale: 3}, 5), far = new Sim().run({scale: 1}, 5);
    expect(palm(near)).toBeCloseTo(palm(far), 6);
    expect(palm(near)).toBeCloseTo(0.07, 6);
  });
});

describe('gestures', () => {
  test('pinch presses at the position held before the fingers closed, and drags', () => {
    const s = new Sim();
    const before = s.run({at: [0.5, 0.5]}, 20);
    const pressed = s.run({pose: 'pinch', at: [0.5, 0.5]}, 4);
    expect(pressed.phase).toBe('press');
    expect(Math.hypot(pressed.x - before.x, pressed.y - before.y)).toBeLessThan(0.01);
    const dragged = s.run({pose: 'pinch', at: [0.4, 0.5]}, 20);
    expect(dragged.down).toBe(true);
    expect(dragged.x - pressed.x).toBeGreaterThan(0.15);
    expect(s.run({at: [0.4, 0.5]}, 3).down).toBe(false);
  });
  test('click: small wobble while pinched and the opening jolt never move the cursor', () => {
    const s = new Sim();
    s.run({at: [0.5, 0.5]}, 20);
    const pressed = s.run({pose: 'pinch', at: [0.5, 0.5]}, 4);
    for (const at of [[0.503, 0.5], [0.497, 0.503], [0.502, 0.498]] as Vec2[]) {
      const wobble = s.run({pose: 'pinch', at}, 3);
      expect([wobble.x, wobble.y]).toEqual([pressed.x, pressed.y]);
    }
    // Opening the fingers shifts the thumb–index midpoint; mouse-up must stay put.
    const released = s.run({at: [0.502, 0.5]});
    expect(released.down).toBe(false);
    expect([released.x, released.y]).toEqual([pressed.x, pressed.y]);
    expect(s.run({at: [0.502, 0.5]}, 2).x).toBe(pressed.x);
    // After the hold, the cursor follows the hand again.
    expect(s.run({at: [0.45, 0.5]}, 10).x).toBeGreaterThan(pressed.x + 0.05);
  });
  test('drag: once past the slop, the cursor follows from the press point without jumping', () => {
    const s = new Sim();
    s.run({at: [0.5, 0.5]}, 20);
    const pressed = s.run({pose: 'pinch', at: [0.5, 0.5]}, 4);
    let prev = pressed.x;
    for (let u = 0.5; u >= 0.45; u -= 0.002) {
      const r = s.run({pose: 'pinch', at: [u, 0.5]});
      expect(r.x - prev).toBeLessThan(0.01); // No jump at the slop boundary.
      prev = r.x;
    }
    expect(prev - pressed.x).toBeGreaterThan(0.06);
  });
  test('touching pads press even though the tip landmarks stay ~1.8 cm apart', () => {
    const s = new Sim();
    s.run({}, 10);
    expect(s.run({pose: 'touch'}, 4).down).toBe(true);
  });
  test('a held pinch stays pressed under landmark noise (no chattering)', () => {
    const s = new Sim(), noise = rng(0.003);
    s.run({noise}, 10);
    s.run({pose: 'touch', noise}, 4);
    for (let i = 0; i < 60; i++) expect(s.run({pose: 'touch', noise}).down).toBe(true);
  });
  test('a thumb crossing in front of the index at another depth does not press', () => {
    const s = new Sim();
    s.run({}, 10);
    expect(s.run({pose: 'cross'}, 10).down).toBe(false);
  });
  test('a hand that appears already pinched must open before it can press', () => {
    const s = new Sim();
    expect(s.run({pose: 'pinch'}, 20).down).toBe(false);
    s.run({}, 3);
    expect(s.run({pose: 'pinch'}, 4).down).toBe(true);
  });
  test('two extended fingers scroll; content follows; each gesture has its own id', () => {
    const s = new Sim();
    s.run({}, 5);
    s.run({pose: 'scroll', at: [0.5, 0.5]}, 6);
    const scroll = s.run({pose: 'scroll', at: [0.5, 0.6]}, 20);
    expect(scroll).toMatchObject({phase: 'scroll', down: false, scroll_id: 1});
    expect(scroll.scroll_y).toBeGreaterThan(3); // Hand moved down 0.2 of the screen.
    s.run({}, 10);
    expect(s.run({pose: 'scroll'}, 10).scroll_id).toBe(2);
  });
  test('only operating poses take control: open palm and fist are idle', () => {
    for (const pose of ['open', 'fist'] as Pose[]) {
      const s = new Sim();
      expect(s.run({pose}, 10)).toMatchObject({phase: 'idle', active: false, down: false});
    }
    const s = new Sim();
    expect(s.run({}, 1).phase).toBe('idle'); // Pointing must hold briefly before it takes over.
    expect(s.run({}, 3)).toMatchObject({phase: 'point', active: true});
    // A fist (thumb over the curled index) never presses, even after an open hand.
    s.run({pose: 'open'}, 10);
    for (let i = 0; i < 10; i++) expect(s.run({pose: 'fist'}).down).toBe(false);
  });
  test('relaxing the hand releases control after a short delay; a flicker does not', () => {
    const s = new Sim();
    s.run({}, 10);
    expect(s.run({pose: 'open'}, 3).phase).toBe('point'); // 100 ms: within the exit delay.
    expect(s.run({}, 1).phase).toBe('point');
    expect(s.run({pose: 'open'}, 10).phase).toBe('idle');
  });
  test('a pinch started from pointing keeps control and presses', () => {
    const s = new Sim();
    s.run({}, 10);
    expect(s.run({pose: 'touch'}, 4)).toMatchObject({phase: 'press', down: true});
  });
  test('a brief scroll-like shape does not trigger scrolling', () => {
    const s = new Sim();
    s.run({}, 5);
    s.run({pose: 'scroll'}, 2);
    expect(s.run({}, 5).phase).toBe('point');
  });
  test('model: idle comes from the model, not from finger-angle rules', () => {
    const s = new Sim();
    // An open palm the model calls pointing points; a pointing hand it calls idle is idle.
    expect(s.run({intent: intent('point'), pose: 'open'}, 5).phase).toBe('point');
    expect(s.run({intent: intent('idle')}, 3)).toMatchObject({phase: 'idle', active: false, down: false});
    expect(s.run({intent: intent('point')}, 3).phase).toBe('point');
    // Below the leave threshold only: 0.5 keeps an idle hand idle (hysteresis).
    s.run({intent: intent('idle')}, 3);
    expect(s.run({intent: [0.5, 0.5, 0, 0]}, 3).phase).toBe('idle');
  });
  test('model: pressing still follows the measured thumb–index gap', () => {
    const s = new Sim();
    s.run({intent: intent('point')}, 5);
    expect(s.run({intent: intent('point'), pose: 'touch'}, 4)).toMatchObject({phase: 'press', down: true});
    expect(s.run({intent: intent('point')}, 1).down).toBe(false);
  });
  test('model: a hand scrolls only once the model calls it a stroke', () => {
    const s = new Sim();
    expect(s.run({intent: intent('point'), pose: 'scroll'}, 10).phase).not.toBe('scroll'); // A return.
    expect(s.run({intent: intent('swipe_down'), pose: 'scroll'}, 1).phase).toBe('scroll'); // No entry delay.
  });
  test('model: a swipe scrolls only in its direction; the return stroke never scrolls back', () => {
    const s = new Sim();
    s.run({intent: intent('point'), pose: 'scroll', at: [0.5, 0.4]}, 5);
    s.run({intent: intent('swipe_down'), pose: 'scroll', at: [0.5, 0.4]}, 1);
    let r!: HandSample;
    for (let y = 0.4; y <= 0.6; y += 0.01) r = s.run({intent: intent('swipe_down'), pose: 'scroll', at: [0.5, y]});
    expect(r).toMatchObject({phase: 'scroll', scroll_id: 1});
    const stroke = r.scroll_y;
    expect(stroke).toBeGreaterThan(5);
    // Hand travels back up while the model still says swipe_down: the total must never shrink
    // (it may still grow a little while the position filter catches up with the stroke).
    let prev = stroke;
    for (let y = 0.6; y >= 0.4; y -= 0.01) {
      r = s.run({intent: intent('swipe_down'), pose: 'scroll', at: [0.5, y]});
      expect(r.scroll_y).toBeGreaterThanOrEqual(prev);
      prev = r.scroll_y;
    }
    expect(r.scroll_y).toBeLessThan(stroke * 1.2);
    expect(r.scroll_id).toBe(1);
  });
  test('output of an older model (other class count) is ignored; rules take over', () => {
    for (const n of [5, 7]) {
      // Index 3 was swipe_up in those models; trusted here it would scroll right away.
      const s = new Sim(), old = Array.from({length: n}, (_, j) => j === 3 ? 0.9 : 0.01);
      expect(s.run({intent: old, pose: 'scroll'}, 2).phase).not.toBe('scroll'); // Rules: entry delay first.
    }
  });
  test('losing the hand releases immediately', () => {
    const s = new Sim();
    s.run({}, 5); s.run({pose: 'pinch'}, 4);
    const gone = s.hand.update({t_ms: s.now += 33, aspect: ASPECT, image: null, world: null}, 2);
    expect(gone).toMatchObject({phase: 'none', active: false, down: false});
  });
  test('One Euro filtering keeps a still hand steady under landmark noise', () => {
    const s = new Sim(), noise = rng(0.002), xs: number[] = [];
    s.run({noise}, 30);
    for (let i = 0; i < 90; i++) xs.push(s.run({noise}).x);
    const mean = xs.reduce((a, b) => a + b) / xs.length;
    const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length);
    const raw = 0.002 / Math.sqrt(3) * 2 / Math.SQRT2; // Unfiltered midpoint jitter, ×2 sensitivity.
    expect(sd).toBeLessThan(raw * 0.5);
  });
});

import {describe, expect, test} from 'bun:test';
import {paint, fromFrames, labelAt, split, removeBoundary, moveBoundary, boundaries, rangeAt, absorbShort, type Segment} from '../ui/annotator/lib/segments';
import {autoLabel} from '../ui/annotator/lib/autolabel';
import type {HandFrame, Vec2, Vec3} from '../ui/shared/hand';

const seg = (start_ms: number, end_ms: number, label: Segment['label']): Segment => ({start_ms, end_ms, label});

describe('timeline painting', () => {
  test('overwrites the middle of a segment, splitting it', () => {
    expect(paint([seg(0, 100, 'point')], 30, 60, 'swipe_up')).toEqual([seg(0, 30, 'point'), seg(30, 60, 'swipe_up'), seg(60, 100, 'point')]);
  });
  test('clears a range across several segments', () => {
    expect(paint([seg(0, 50, 'point'), seg(50, 100, 'idle')], 40, 70, null)).toEqual([seg(0, 40, 'point'), seg(70, 100, 'idle')]);
  });
  test('merges touching segments with the same label', () => {
    expect(paint([seg(0, 50, 'point'), seg(50, 100, 'idle')], 50, 100, 'point')).toEqual([seg(0, 100, 'point')]);
  });
  test('split keeps both halves; relabelling one half leaves the cut', () => {
    const cut = split([seg(0, 100, 'point')], 40);
    expect(cut).toEqual([seg(0, 40, 'point'), seg(40, 100, 'point')]);
    expect(paint(cut, 40, 100, 'swipe_up')).toEqual([seg(0, 40, 'point'), seg(40, 100, 'swipe_up')]);
    // A paint elsewhere does not silently undo an unrelated split.
    expect(paint([...cut, seg(100, 150, 'idle')], 100, 150, 'swipe_up')).toEqual([seg(0, 40, 'point'), seg(40, 100, 'point'), seg(100, 150, 'swipe_up')]);
    expect(split([seg(0, 100, 'point')], 5)).toEqual([seg(0, 100, 'point')]); // Too close to an edge.
  });
  test('removing a cut joins neighbours (left label) or extends across a gap', () => {
    expect(removeBoundary([seg(0, 50, 'point'), seg(50, 100, 'swipe_up')], 50, 200)).toEqual([seg(0, 100, 'point')]);
    expect(removeBoundary([seg(0, 50, 'point'), seg(80, 100, 'idle')], 50, 200)).toEqual([seg(0, 80, 'point'), seg(80, 100, 'idle')]);
    expect(removeBoundary([seg(80, 100, 'idle')], 80, 200)).toEqual([seg(0, 100, 'idle')]);
    expect(removeBoundary([seg(0, 50, 'point')], 50, 200)).toEqual([seg(0, 200, 'point')]);
  });
  test('moving a cut adjusts both sides and never crosses a neighbour', () => {
    const s = [seg(0, 50, 'point'), seg(50, 100, 'swipe_up'), seg(120, 150, 'idle')];
    expect(moveBoundary(s, 50, 70, 200)).toEqual([seg(0, 70, 'point'), seg(70, 100, 'swipe_up'), seg(120, 150, 'idle')]);
    expect(moveBoundary(s, 50, 500, 200)[0].end_ms).toBe(80); // Clamped to leave MIN_MS.
    expect(moveBoundary(s, 100, 140, 200)[1].end_ms).toBe(120); // Stops at the next segment.
    expect(boundaries(s)).toEqual([0, 50, 100, 120, 150]);
  });
  test('short fragments fold into the longer touching neighbour; isolated ones stay', () => {
    expect(absorbShort([seg(0, 500, 'point'), seg(500, 560, 'swipe_up'), seg(560, 800, 'idle')], 150))
      .toEqual([seg(0, 560, 'point'), seg(560, 800, 'idle')]);
    // Folding an A between two Bs joins them into one.
    expect(absorbShort([seg(0, 300, 'idle'), seg(300, 340, 'swipe_down'), seg(340, 600, 'idle')], 150)).toEqual([seg(0, 600, 'idle')]);
    expect(absorbShort([seg(0, 300, 'idle'), seg(400, 420, 'swipe_up')], 150)).toEqual([seg(0, 300, 'idle'), seg(400, 420, 'swipe_up')]);
  });
  test('clicking a gap selects the whole gap', () => {
    const s = [seg(0, 50, 'point'), seg(80, 100, 'idle')];
    expect(rangeAt(s, 60, 200)).toEqual([50, 80]);
    expect(rangeAt(s, 150, 200)).toEqual([100, 200]);
    expect(rangeAt(s, 10, 200)).toEqual([0, 50]);
  });
  test('frames → segments with gaps where there is no hand', () => {
    const s = fromFrames([0, 33, 66, 99, 132], ['point', 'point', null, 'swipe_up', 'swipe_up']);
    expect(s).toEqual([seg(0, 66, 'point'), seg(99, 165, 'swipe_up')]);
    expect(labelAt(s, 70)).toBeNull();
    expect(labelAt(s, 100)).toBe('swipe_up');
  });
});

function hand(t_ms: number, x: number, y: number): HandFrame {
  const world: Vec3[] = Array.from({length: 21}, (_, j) => [j * 0.003, -j * 0.004, 0]);
  world[4] = [0.08, -0.06, 0]; // Open thumb.
  return {t_ms, aspect: 1, world, image: world.map(([wx, wy]): Vec2 => [x + wx, y + wy])};
}

describe('rule + prompt auto-labelling', () => {
  test('an idle take is idle wherever a hand is visible', () => {
    const frames = [hand(0, 0.5, 0.5), {t_ms: 33, aspect: 1, image: null, world: null}, hand(66, 0.5, 0.5)];
    expect(autoLabel(frames, 'idle').map(s => s.label)).toEqual(['idle', 'idle']);
  });
  test('in a swipe take, fast prompted strokes are the swipe and slow returns are idle', () => {
    const frames: HandFrame[] = [];
    let y = 0.3, t = 0;
    for (let cycle = 0; cycle < 3; cycle++) {
      for (let i = 0; i < 10; i++) { y += 0.012; frames.push(hand(t += 33, 0.5, y)); } // Down fast.
      for (let i = 0; i < 30; i++) { y -= 0.004; frames.push(hand(t += 33, 0.5, y)); } // Back slowly.
    }
    const s = autoLabel(frames, 'swipe_down');
    expect(s.filter(x => x.label === 'swipe_down').length).toBe(3);
    expect(s.every(x => x.label === 'swipe_down' || x.label === 'idle')).toBe(true);
  });
  test('in a point take, the pointing pose is point and any other hand is idle', () => {
    // Index straight; middle, ring and pinky straight only when the palm is open.
    const posed = (t_ms: number, open: boolean): HandFrame => {
      const world: Vec3[] = Array.from({length: 21}, () => [0, 0, 0]);
      [5, 9, 13, 17].forEach((mcp, f) => {
        const x = 0.03 - f * 0.02, straight = f === 0 || open;
        world[mcp] = [x, -0.08, 0]; world[mcp + 1] = [x, -0.11, 0];
        world[mcp + 2] = straight ? [x, -0.13, 0] : [x, -0.1, 0.02];
        world[mcp + 3] = straight ? [x, -0.15, 0] : [x, -0.08, 0.02];
      });
      world[1] = [0.03, -0.02, 0]; world[2] = [0.05, -0.04, 0]; world[3] = [0.06, -0.06, 0]; world[4] = [0.075, -0.09, 0];
      return {t_ms, aspect: 1, world, image: world.map(([wx, wy]): Vec2 => [0.5 + wx, 0.5 + wy])};
    };
    const frames = Array.from({length: 60}, (_, i) => posed(i * 33, i >= 20 && i < 40));
    // The 80 ms entry delay at the start is shorter than a fragment and is folded away.
    expect(autoLabel(frames, 'point').map(s => s.label)).toEqual(['point', 'idle', 'point']);
  });
});

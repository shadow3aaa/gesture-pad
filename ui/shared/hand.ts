import {LABELS} from './labels';

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
/** One detected hand: 21 landmarks normalised to the camera image (0–1), and metric shape. */
export interface HandFrame {t_ms: number; aspect: number; image: Vec2[] | null; world: Vec3[] | null}

/** What the overlay draws and the backend executes. Screen coordinates are fractions of the display. */
export interface HandSample {
  phase: 'none' | 'idle' | 'point' | 'press' | 'scroll';
  active: boolean;
  down: boolean;
  x: number;
  y: number;
  skeleton: Vec2[] | null; // Screen-space hand drawn around the action point.
  scroll_id: number; // Non-zero while scrolling; changes for each new scroll gesture.
  scroll_y: number; // Cumulative vertical wheel notches since this scroll gesture started.
  pinch: number | null; // Thumb–index gap as seen on screen, in hand lengths; compare with PINCH.
}
export const none: HandSample = {phase: 'none', active: false, down: false, x: 0, y: 0, skeleton: null, scroll_id: 0, scroll_y: 0, pinch: null};

export const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
export const DEFAULT_SENSITIVITY = 2;
/**
 * Pinch thresholds, in hand lengths (wrist → middle knuckle, ≈ 9 cm). The visible (2-D)
 * gap decides, so the click matches what the drawn skeleton shows. MediaPipe's tip
 * landmarks sit at the nail ends, so touching pads still leave ≈ 1.5–2 cm between them.
 * The 3-D gap only vetoes a thumb that crosses in front of the index at a different depth.
 */
export const PINCH = {press: 0.3, release: 0.42, depthVeto: 0.55};
const PRESS_MS = 40, REWIND_MS = 100;
/**
 * Tap slop: after a press the cursor holds still until the hand moves this far (screen
 * fraction), so pinch jitter never turns a click into a tiny drag. After a release the
 * cursor holds for a moment too, so opening fingers do not move the mouse-up point.
 */
const TAP_SLOP = 0.012, RELEASE_HOLD_MS = 150, RELEASE_ESCAPE = 0.04;
const SCROLL_ENTER_MS = 120, SCROLL_EXIT_MS = 100, NOTCHES_PER_SCREEN = 25;
/** Rule mode: an operating pose must hold this long to take control, and be gone this long to release it. */
const ENGAGE_ENTER_MS = 80, ENGAGE_EXIT_MS = 200;
const HAND_SIZE = 0.07; // Drawn palm width, as a fraction of screen width.
/** Positions in the model output (LABELS order) of the classes the controller reads. */
const IDLE = LABELS.indexOf('idle'), UP = LABELS.indexOf('swipe_up'), DOWN = LABELS.indexOf('swipe_down');

const dist3 = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const clamp = (v: number) => Math.max(0, Math.min(1, v));

/**
 * Absolute mapping: the camera image, mirrored and scaled about its centre, covers the
 * screen. Sensitivity 2 means the middle half of the image spans the whole screen.
 */
export const toScreen = ([u, v]: Vec2, sensitivity: number): Vec2 => [0.5 - (u - 0.5) * sensitivity, 0.5 + (v - 0.5) * sensitivity];

/** Causal One Euro filter. */
class Filter {
  private value: number | null = null;
  private raw = 0;
  private velocity = 0;
  constructor(private cutoff: number, private beta: number) {}
  reset() { this.value = null; }
  update(value: number, dt: number) {
    const alpha = (c: number) => 1 / (1 + 1 / (2 * Math.PI * c * dt));
    if (this.value === null) { this.value = this.raw = value; this.velocity = 0; return value; }
    this.velocity += alpha(1) * ((value - this.raw) / dt - this.velocity);
    this.raw = value;
    this.value += alpha(this.cutoff + this.beta * Math.abs(this.velocity)) * (value - this.value);
    return this.value;
  }
}

/** Fingertip farther from the wrist than its middle joint: the finger is straight. */
const extended = (w: Vec3[], tip: number, pip: number) => dist3(w[tip], w[0]) > dist3(w[pip], w[0]) * 1.15;

/**
 * The user's hand is the cursor. No calibration and no eye tracking: the skeleton is
 * shown on screen and the user corrects by sight, so mapping only has to be stable.
 * Operating poses: index alone extended points (thumb free); thumb onto index presses;
 * index + middle extended scrolls. Any other hand (open palm, fist, relaxed) is idle.
 */
export class HandControl {
  private filters = Array.from({length: 42}, () => new Filter(1, 8));
  private last: number | null = null;
  private down = false;
  private releasedSinceSeen = false;
  private pressCandidate: number | null = null;
  private press: {anchor: Vec2; start: Vec2; dragging: boolean} | null = null;
  private hold: {at: Vec2; until: number; from: Vec2} | null = null;
  private lastOut: Vec2 | null = null;
  private history: {t: number; p: Vec2}[] = [];
  private scrollPose: {since: number; lost: number | null} | null = null;
  private engagePose: {since: number; lost: number | null} | null = null;
  private scroll: {id: number; origin: number; dir: -1 | 1 | null; total: number; last: number} | null = null;
  private scrollId = 0;
  private idle = false;
  /** Screen-y direction of the swipe the model recognises (−1 up, 1 down). */
  private swipe: -1 | 1 | null = null;

  reset() {
    this.filters.forEach(f => f.reset());
    this.last = this.pressCandidate = this.press = this.hold = this.lastOut = this.scrollPose = this.engagePose = this.scroll = this.swipe = null;
    this.down = this.releasedSinceSeen = this.idle = false;
    this.history = [];
  }

  /**
   * `intent`: class probabilities from the GCN (order of LABELS in labels.ts), or null to use
   * rules. The model decides whether the user is idle and whether (and which way) they
   * swipe; pressing, where the cursor goes, tap slop and release hold stay deterministic.
   */
  update(frame: HandFrame, sensitivity: number, intent: readonly number[] | null = null): HandSample {
    const {image, world, t_ms: now} = frame;
    const valid = image?.length === 21 && world?.length === 21 && [...image, ...world].every(p => p.every(Number.isFinite));
    if (!valid || !Number.isFinite(sensitivity) || sensitivity <= 0 || (this.last !== null && now - this.last > 250)) {
      this.reset();
      if (!valid) return none;
    }
    const dt = this.last === null || now <= this.last ? 1 / 30 : (now - this.last) / 1000;
    this.last = now;
    const p = image!.map(([u, v], i): Vec2 => [this.filters[2 * i].update(u, dt), this.filters[2 * i + 1].update(v, dt)]);
    const w = world!;
    // Thumb tip to the index tip or its last joint: pads touch between those points.
    // Unfiltered landmarks: filter lag would delay the click; the hysteresis band absorbs noise.
    const aspect = Number.isFinite(frame.aspect) && frame.aspect > 0 ? frame.aspect : 16 / 9;
    const q = image!;
    const d2 = (i: number, j: number) => Math.hypot((q[i][0] - q[j][0]) * aspect, q[i][1] - q[j][1]);
    const pinch = Math.min(d2(4, 8), d2(4, 7)) / (d2(0, 9) || 1);
    const depthGap = Math.min(dist3(w[4], w[8]), dist3(w[4], w[7])) / dist3(w[0], w[9]);
    // Press and release always come from the measured gap, also with a model: the rule is
    // exact and calibrated, while a model blurs the frames where fingers close or open.
    const closed = pinch < PINCH.press && depthGap < PINCH.depthVeto;
    const open = pinch > PINCH.release;
    // Hand shape, with entry and exit delays so a passing finger shape does not toggle anything.
    const [index, middle, ring, pinky] = [[8, 6], [12, 10], [16, 14], [20, 18]].map(([tip, pip]) => extended(w, tip, pip));
    const twoFingers = index && middle && !ring && !pinky;
    const scrollShape = twoFingers && open && !this.down;
    if (scrollShape) { this.scrollPose ??= {since: now, lost: null}; this.scrollPose.lost = null; }
    else if (this.scrollPose) { this.scrollPose.lost ??= now; if (now - this.scrollPose.lost > SCROLL_EXIT_MS) this.scrollPose = null; }
    // Operating poses: pointing (index alone, thumb free), pinching or the scroll shape.
    // A pinching index may bend onto the thumb, but its tip stays out past the knuckle;
    // in a fist it folds back into the palm.
    const indexOut = dist3(w[8], w[0]) > dist3(w[5], w[0]) * 1.1;
    const engaged = (index && !middle && !ring && !pinky) || (!open && indexOut && !ring && !pinky) || twoFingers;
    if (engaged) { this.engagePose ??= {since: now, lost: null}; this.engagePose.lost = null; }
    else if (this.engagePose) { this.engagePose.lost ??= now; if (now - this.engagePose.lost > ENGAGE_EXIT_MS) this.engagePose = null; }

    let scrolling: boolean;
    if (intent && intent.length === LABELS.length && intent.every(Number.isFinite)) {
      // The model decides idle and swipes (hysteresis: enter above 0.6, leave below 0.4). Its
      // idle class is everything that is not an operation, learnt from data rather than from
      // finger-angle thresholds.
      this.idle = this.idle ? intent[IDLE] > 0.4 : intent[IDLE] > 0.6;
      const [best, dir] = intent[DOWN] > intent[UP] ? [DOWN, 1 as const] : [UP, -1 as const];
      if (this.swipe !== null && intent[this.swipe === 1 ? DOWN : UP] < 0.4) this.swipe = null;
      if (intent[best] > 0.6) this.swipe = dir;
      scrolling = this.swipe !== null && !this.down;
    } else {
      this.swipe = null;
      scrolling = !!this.scrollPose && now - this.scrollPose.since >= SCROLL_ENTER_MS;
      // Without a model, idle is any hand shape other than pointing, pinching or scrolling.
      this.idle = !scrolling && !(this.engagePose && now - this.engagePose.since >= ENGAGE_ENTER_MS);
    }

    // Action point: between thumb and index tip (stable while pinching), or between the two scroll fingers.
    const [a, b] = scrolling ? [8, 12] : [4, 8];
    const action = toScreen([(p[a][0] + p[b][0]) / 2, (p[a][1] + p[b][1]) / 2], sensitivity);
    const palm = Math.hypot(p[5][0] - p[17][0], p[5][1] - p[17][1]) || 1;
    const centre: Vec2 = [(p[a][0] + p[b][0]) / 2, (p[a][1] + p[b][1]) / 2];
    // Drawn at a fixed size around the action point; shape is exact, scale is not.
    const around = ([x, y]: Vec2) => p.map(([u, v]): Vec2 => [x - (u - centre[0]) / palm * HAND_SIZE, y + (v - centre[1]) / palm * HAND_SIZE]);
    const base = {...none, active: true, skeleton: around(action), pinch, x: clamp(action[0]), y: clamp(action[1])};

    if (this.idle && !this.down) {
      // Not operating: the cursor stays where it is and nothing is pressed.
      this.pressCandidate = this.press = this.hold = this.scroll = null;
      return {...base, phase: 'idle', active: false};
    }
    if (scrolling) {
      this.pressCandidate = this.press = null;
      // A new gesture, or the model changing direction, starts a new scroll.
      // Vertical only: sideways swipes lose hand tracking too often to be usable.
      const dir = this.swipe, y = action[1];
      if (!this.scroll || this.scroll.dir !== dir) this.scroll = {id: ++this.scrollId, origin: y, dir, total: 0, last: y};
      const s = this.scroll;
      let dy = y - s.origin;
      if (s.dir) {
        // Model mode is a ratchet: only motion in the recognised direction scrolls, so the
        // return stroke never scrolls back.
        s.total += Math.max(0, (y - s.last) * s.dir);
        dy = s.dir * s.total;
      }
      s.last = y;
      // Content follows the fingers, like a touchpad.
      return {...base, phase: 'scroll', scroll_id: s.id, scroll_y: dy * NOTCHES_PER_SCREEN};
    }
    this.scroll = null;

    this.history.push({t: now, p: action});
    while (this.history.length > 1 && now - this.history[0].t > 300) this.history.shift();
    // Pinch with hysteresis and debounce. A hand that appears already pinched must open first.
    if (open) {
      if (this.down && this.lastOut) this.hold = {at: this.lastOut, until: now + RELEASE_HOLD_MS, from: action};
      this.releasedSinceSeen = true; this.down = false; this.press = this.pressCandidate = null;
    } else if (!this.down && closed && this.releasedSinceSeen) {
      this.pressCandidate ??= now;
      if (now - this.pressCandidate >= PRESS_MS) {
        // Anchor to where the hand was just before the fingers started closing.
        const before = this.history.find(h => h.t >= this.pressCandidate! - REWIND_MS) ?? this.history[0];
        this.press = {anchor: before.p, start: action, dragging: false};
        this.hold = null;
        this.down = true;
      }
    } else if (!this.down) this.pressCandidate = null;

    let out = action;
    if (this.press) {
      const {anchor, start} = this.press;
      // Within the tap slop the cursor stays put; past it, dragging measures from the slop
      // boundary, so it lags only by the slop and never jumps.
      const moved = Math.hypot(action[0] - start[0], action[1] - start[1]);
      if (!this.press.dragging && moved > TAP_SLOP) {
        const k = TAP_SLOP / moved;
        this.press.dragging = true;
        this.press.start = [start[0] + (action[0] - start[0]) * k, start[1] + (action[1] - start[1]) * k];
      }
      out = this.press.dragging ? [anchor[0] + action[0] - this.press.start[0], anchor[1] + action[1] - this.press.start[1]] : anchor;
    } else if (this.hold) {
      // Hold where the button went up, unless the hand clearly moves on.
      if (now < this.hold.until && Math.hypot(action[0] - this.hold.from[0], action[1] - this.hold.from[1]) < RELEASE_ESCAPE) out = this.hold.at;
      else this.hold = null;
    }
    const at: Vec2 = [clamp(out[0]), clamp(out[1])];
    this.lastOut = at;
    return {...base, phase: this.down ? 'press' : 'point', down: this.down, x: at[0], y: at[1], skeleton: around(at)};
  }
}

/** Skeleton colour per state, shared by the camera preview and the screen overlay (matches labels.ts). */
export const PHASE_COLOR: Record<HandSample['phase'], string> = {none: '#94a3b8', idle: '#94a3b8', point: '#22c55e', press: '#f59e0b', scroll: '#38bdf8'};

/** Draws the hand overlay. Shared by the desktop overlay window and the in-app preview. */
export function drawHand(ctx: CanvasRenderingContext2D, width: number, height: number, s: HandSample) {
  ctx.clearRect(0, 0, width, height);
  if (!s.skeleton) return;
  const at = ([x, y]: Vec2): Vec2 => [x * width, y * height];
  const accent = PHASE_COLOR[s.phase];
  const unit = Math.max(1.5, Math.min(width, height) / 500);
  ctx.globalAlpha = s.phase === 'idle' ? 0.35 : 1; // Idle: visible but clearly not in control.
  ctx.lineCap = 'round';
  ctx.lineWidth = unit * 2;
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  ctx.shadowColor = 'rgba(0,0,0,0.6)';
  ctx.shadowBlur = unit * 3;
  ctx.beginPath();
  for (const [a, b] of EDGES) { ctx.moveTo(...at(s.skeleton[a])); ctx.lineTo(...at(s.skeleton[b])); }
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  for (const q of s.skeleton) { ctx.beginPath(); ctx.arc(...at(q), unit * 2.5, 0, Math.PI * 2); ctx.fill(); }
  // The action point: where clicks and drags land.
  const [x, y] = at([s.x, s.y]);
  ctx.shadowBlur = 0;
  ctx.lineWidth = unit * 2;
  ctx.strokeStyle = accent;
  ctx.beginPath(); ctx.arc(x, y, unit * 7, 0, Math.PI * 2); ctx.stroke();
  if (s.down || s.phase === 'scroll') { ctx.fillStyle = accent; ctx.beginPath(); ctx.arc(x, y, unit * 4, 0, Math.PI * 2); ctx.fill(); }
  ctx.globalAlpha = 1;
}

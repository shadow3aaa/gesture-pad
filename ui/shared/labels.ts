/**
 * Intent classes, in model output order; must match `LABELS` in crates/gesture-model.
 * Pressing is decided by rules (thumb–index gap), so a pinching hand is `point`.
 */
export const LABELS = ['idle', 'point', 'swipe_up', 'swipe_down'] as const;
export type Label = (typeof LABELS)[number];

export const META: Record<Label, {name: string; color: string; prompt: string}> = {
  idle: {name: '空闲', color: '#94a3b8', prompt: '不操作时手的各种样子：张开手掌、握拳、手放松、拿东西、说话时比划。别做指向和双指手势。'},
  point: {name: '指向', color: '#22c55e', prompt: '只伸食指，其余三指收起，拇指随意。在屏幕上到处移动，中间穿插点击和拖动。'},
  swipe_up: {name: '上划', color: '#38bdf8', prompt: '伸出食指和中指，照预览画面往上划，划完放松收回。重复。'},
  swipe_down: {name: '下划', color: '#6366f1', prompt: '伸出食指和中指，照预览画面往下划，划完放松收回。重复。'},
};

/** Prompts of takes recorded with labels that were since removed or merged. */
const OLD_PROMPTS: Record<string, string> = {pinch: '捏合（已并入指向）', swipe_left: '左划（已移除）', swipe_right: '右划（已移除）'};

/** Display info for a recording prompt, including prompts of labels that were since removed. */
export const promptMeta = (prompt: string) =>
  (META as Record<string, (typeof META)[Label] | undefined>)[prompt] ?? {name: OLD_PROMPTS[prompt] ?? prompt, color: '#cbd5e1', prompt: ''};

/**
 * Screen-space direction of each swipe (finger motion as seen in the mirrored preview).
 * Only vertical: fast sideways swipes lose hand tracking too often to be usable.
 */
export const SWIPE: Partial<Record<Label, [number, number]>> = {swipe_up: [0, -1], swipe_down: [0, 1]};

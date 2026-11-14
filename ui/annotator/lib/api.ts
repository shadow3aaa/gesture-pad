import {invoke, isTauri} from '@tauri-apps/api/core';
import type {HandFrame} from '@/hand';
import type {Label} from '@/labels';
import type {Segment} from './segments';

/**
 * Mirrors `Session` in crates/gesture-model. Landmarks only, no video. `prompt` may be a
 * label that has since been removed (old left/right swipe takes); use `promptMeta` to show it.
 */
export interface Session {version: 1; id: string; prompt: Label | string; reviewed: boolean; frames: HandFrame[]; segments: Segment[]}
/** Mirrors `EpochStats` in crates/gesture-model/src/train.rs. */
export interface EpochStats {
  epoch: number; epochs: number; lr: number;
  /** `*_balanced`: mean recall over classes that have samples (every class counts the same). */
  train_loss: number; train_balanced: number; val_loss: number; val_balanced: number;
  class_recall: (number | null)[]; best: boolean;
  /** Epochs since the best one; training ends early when it reaches `patience`. */
  stale: number; patience: number; elapsed_ms: number;
}
/** Mirrors `Progress`: one `start`, then throttled `batch` and one `epoch` per epoch. */
export type Progress =
  | {kind: 'start'; labels: Label[]; epochs: number; train_counts: number[]; val_counts: number[]}
  | {kind: 'batch'; epoch: number; batch: number; batches: number; loss: number}
  | ({kind: 'epoch'} & EpochStats);
export interface Report {labels: Label[]; train_samples: number; val_samples: number; best_epoch: number; val_balanced: number; confusion: number[][]; stopped: boolean; early_stopped: boolean}

export const desktop = isTauri();
export const listSessions = (root: string) => invoke<Session[]>('list_sessions', {root});
export const saveSession = (root: string, session: Session) => invoke('save_session', {root, session});
export const deleteSession = (root: string, id: string) => invoke('delete_session', {root, id});
export const trainModel = (root: string, epochs: number) => invoke<Report>('train_model', {root, epochs});
export const stopTraining = () => invoke('stop_training');

export const newId = () => `${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;

/** Trim landmark precision: sessions stay small and diff-friendly, far below MediaPipe's noise. */
export const compact = (f: HandFrame): HandFrame => {
  const r = (v: number) => Math.round(v * 1e5) / 1e5;
  return {
    t_ms: Math.round(f.t_ms * 10) / 10, aspect: f.aspect,
    image: f.image?.map(([u, v]) => [r(u), r(v)]) ?? null,
    world: f.world?.map(([x, y, z]) => [r(x), r(y), r(z)]) ?? null,
  };
};

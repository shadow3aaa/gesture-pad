import {FilesetResolver, HandLandmarker} from '@mediapipe/tasks-vision';
import {EDGES, PHASE_COLOR, type HandFrame, type HandSample, type Vec2, type Vec3} from './hand';

export async function detector() {
  const files = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  return HandLandmarker.createFromOptions(files, {
    baseOptions: {modelAssetPath: '/mediapipe/hand_landmarker.task', delegate: 'GPU'},
    runningMode: 'VIDEO', numHands: 1, minHandDetectionConfidence: 0.6, minTrackingConfidence: 0.6,
  });
}

export function detect(model: HandLandmarker, video: HTMLVideoElement, t_ms: number): HandFrame {
  const r = model.detectForVideo(video, t_ms);
  return {
    t_ms,
    aspect: video.videoWidth / video.videoHeight,
    image: r.landmarks[0]?.map(p => [p.x, p.y] as Vec2) ?? null,
    world: r.worldLandmarks[0]?.map(p => [p.x, p.y, p.z] as Vec3) ?? null,
  };
}

/** Camera preview overlay: the detected hand, coloured by state. */
export function paint(canvas: HTMLCanvasElement, frame: HandFrame, sample: HandSample) {
  const ctx = canvas.getContext('2d')!;
  const {width: w, height: h} = canvas;
  ctx.clearRect(0, 0, w, h);
  if (!frame.image) return;
  const p = frame.image.map(([u, v]): Vec2 => [u * w, v * h]);
  ctx.strokeStyle = ctx.fillStyle = PHASE_COLOR[sample.phase];
  ctx.lineWidth = Math.max(2, w / 400);
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const [a, b] of EDGES) { ctx.moveTo(...p[a]); ctx.lineTo(...p[b]); }
  ctx.stroke();
  for (const q of p) { ctx.beginPath(); ctx.arc(q[0], q[1], Math.max(3, w / 300), 0, Math.PI * 2); ctx.fill(); }
}

export async function openCamera(video: HTMLVideoElement, signal: AbortSignal) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {width: {ideal: 1280}, height: {ideal: 720}, frameRate: {ideal: 30}}, audio: false,
  });
  const stop = () => { stream.getTracks().forEach(t => t.stop()); };
  if (signal.aborted) { stop(); throw new DOMException('已取消', 'AbortError'); }
  signal.addEventListener('abort', stop, {once: true});
  try {
    await new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timer);
        video.removeEventListener('loadeddata', ready);
        video.removeEventListener('error', fail);
        signal.removeEventListener('abort', cancel);
      };
      const ready = () => { cleanup(); resolve(); };
      const fail = () => { cleanup(); reject(new Error('摄像头画面无法解码')); };
      const cancel = () => { cleanup(); reject(new DOMException('已取消', 'AbortError')); };
      const timer = setTimeout(() => { cleanup(); reject(new Error('摄像头启动超时，请检查设备是否被其他程序占用')); }, 12000);
      video.addEventListener('loadeddata', ready, {once: true});
      video.addEventListener('error', fail, {once: true});
      signal.addEventListener('abort', cancel, {once: true});
      video.srcObject = stream;
    });
    await video.play();
    return stream;
  } catch (error) {
    stop();
    signal.removeEventListener('abort', stop);
    throw error;
  }
}

export function cameraError(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError') return '摄像头权限被拒绝。请在系统和浏览器设置中允许访问摄像头后重试。';
    if (error.name === 'NotFoundError') return '未找到摄像头，请连接设备后重试。';
    if (error.name === 'NotReadableError') return '摄像头无法打开，可能被其他程序占用。';
  }
  return error instanceof Error ? error.message : String(error);
}

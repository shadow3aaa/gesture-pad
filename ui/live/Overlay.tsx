import {useEffect, useRef} from 'react';
import {listen} from '@tauri-apps/api/event';
import {drawHand, type HandSample} from '@/hand';

/** Transparent, click-through, always-on-top window that draws the hand over the desktop. */
export function Overlay() {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    document.documentElement.style.background = document.body.style.background = 'transparent';
    const c = canvas.current!, ctx = c.getContext('2d')!;
    const resize = () => {
      c.width = innerWidth * devicePixelRatio;
      c.height = innerHeight * devicePixelRatio;
    };
    resize();
    addEventListener('resize', resize);
    // Clear if the stream stops, so a stale hand never lingers on screen.
    let timer = 0;
    const unlisten = listen<HandSample>('hand', ({payload}) => {
      drawHand(ctx, c.width, c.height, payload);
      clearTimeout(timer);
      timer = window.setTimeout(() => ctx.clearRect(0, 0, c.width, c.height), 300);
    });
    return () => { removeEventListener('resize', resize); clearTimeout(timer); void unlisten.then(f => f()); };
  }, []);
  return <canvas ref={canvas} className="pointer-events-none fixed inset-0 size-full" />;
}

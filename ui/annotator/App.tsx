import {lazy, Suspense, useEffect, useState} from 'react';
import {ActivityIcon, CircleIcon, FolderOpenIcon, MoonIcon, SunIcon, TagsIcon} from 'lucide-react';
import {toast} from 'sonner';
import {AsyncButton, errorText} from '@/components/async-button';
import {Badge} from '@/components/ui/badge';
import {Button} from '@/components/ui/button';
import {Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle} from '@/components/ui/empty';
import {ScrollArea} from '@/components/ui/scroll-area';
import {promptMeta} from '@/labels';
import {cn} from '@/lib/utils';
import {useTheme} from '@/theme';
import {deleteSession, desktop, listSessions, saveSession, type Session} from './lib/api';
import {Recorder} from './components/Recorder';
import {Review} from './components/Review';
// Charts are heavy; load the training dialog the first time it opens.
const TrainDialog = lazy(() => import('./components/TrainDialog').then(m => ({default: m.TrainDialog})));

export default function App() {
  const {theme, toggleTheme} = useTheme();
  const [root, setRoot] = useState(() => localStorage.getItem('dataset-root') ?? '');
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<Session | null>(null);
  const [recording, setRecording] = useState(false);
  const [training, setTraining] = useState(false);
  // Stays mounted after the first open, so results survive closing the dialog.
  const [trainOpened, setTrainOpened] = useState(false);
  useEffect(() => { if (training) setTrainOpened(true); }, [training]);
  const dirty = !!draft && draft !== sessions.find(s => s.id === draft.id);

  const refresh = async (dir = root) => { if (desktop && dir) setSessions(await listSessions(dir)); };
  useEffect(() => { refresh().catch(e => toast.error(errorText(e))); }, []);

  const discardOk = () => !dirty || confirm('当前会话有未保存的修改，放弃？');
  const open = (s: Session) => { if (discardOk()) { setRecording(false); setSelected(s.id); setDraft(s); } };
  const chooseRoot = async () => {
    if (!discardOk()) return;
    const {open: pick} = await import('@tauri-apps/plugin-dialog');
    const dir = await pick({directory: true, title: '选择数据目录'});
    if (typeof dir !== 'string') return;
    await refresh(dir);
    setRoot(dir);
    localStorage.setItem('dataset-root', dir);
    setSelected(null); setDraft(null);
  };
  const persist = async (s: Session) => {
    if (desktop) await saveSession(root, s);
    setSessions(list => [...list.filter(x => x.id !== s.id), s].sort((a, b) => a.id.localeCompare(b.id)));
  };
  const save = async () => { if (draft) { await persist(draft); toast.success('已保存'); } };
  const remove = async () => {
    if (!draft || !confirm('删除这个会话？')) return;
    if (desktop) await deleteSession(root, draft.id);
    setSessions(list => list.filter(s => s.id !== draft.id));
    setSelected(null); setDraft(null);
  };
  const recorded = async (s: Session) => {
    await persist(s);
    setRecording(false); setSelected(s.id); setDraft(s);
    toast.success('录好了，标签已自动生成，检查一下');
  };

  const reviewed = sessions.filter(s => s.reviewed).length;
  const ready = !desktop || !!root;
  return (
    <div className="grid h-full grid-cols-[17rem_minmax(0,1fr)]">
      <aside className="flex min-h-0 flex-col border-r bg-sidebar">
        <div className="flex h-14 items-center gap-2 border-b px-3">
          <AsyncButton variant="ghost" className="min-w-0 flex-1 justify-start" onClick={chooseRoot} disabled={!desktop} title={root || undefined}>
            <FolderOpenIcon data-icon="inline-start" /><span className="truncate">{root ? root.split(/[\\/]/).filter(Boolean).pop() : desktop ? '选择数据目录' : '浏览器预览'}</span>
          </AsyncButton>
          <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="切换主题">{theme === 'light' ? <MoonIcon /> : <SunIcon />}</Button>
        </div>
        <div className="grid grid-cols-2 gap-2 p-3">
          <Button disabled={!ready} onClick={() => { if (discardOk()) { setRecording(true); setSelected(null); setDraft(null); } }}><CircleIcon data-icon="inline-start" className="fill-current" />采集</Button>
          <Button variant="outline" disabled={!ready} onClick={() => setTraining(true)}><ActivityIcon data-icon="inline-start" />训练</Button>
        </div>
        <p className="px-3 pb-2 text-xs text-muted-foreground">{sessions.length} 个会话 · {reviewed} 个已校对</p>
        <ScrollArea className="min-h-0 flex-1">
          <div className="flex flex-col gap-1 px-2 pb-3">
            {sessions.map(s => (
              <button key={s.id} type="button" onClick={() => open(s)} className={cn('flex items-center gap-2 rounded-md px-2 py-2 text-left text-sm hover:bg-sidebar-accent', s.id === selected && 'bg-sidebar-accent')}>
                <span className="size-2.5 shrink-0 rounded-full" style={{background: promptMeta(s.prompt).color}} />
                <span className="flex-1 truncate">{promptMeta(s.prompt).name} · {((s.frames.at(-1)?.t_ms ?? 0) / 1000).toFixed(0)}s</span>
                <Badge variant={s.reviewed ? 'default' : 'outline'}>{s.reviewed ? '已校对' : '待校对'}</Badge>
              </button>
            ))}
          </div>
        </ScrollArea>
      </aside>
      <main className="flex min-h-0 min-w-0 flex-col p-5">
        {recording ? <Recorder onRecorded={recorded} />
          : draft ? <Review session={draft} dirty={dirty} onChange={setDraft} onSave={save} onDelete={remove} />
          : (
            <Empty>
              <EmptyHeader>
                <EmptyMedia variant="icon"><TagsIcon /></EmptyMedia>
                <EmptyTitle>{ready ? '采集、校对、训练' : '先选择数据目录'}</EmptyTitle>
                <EmptyDescription>{ready
                  ? '先采集几段手势，在时间轴上改对标签，然后训练。'
                  : '录下的会话存在这个目录里，训练出的模型放在它的 model 子目录。'}{!desktop && ' 浏览器里录的会话不会保存，也不能训练。'}</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
      </main>
      {(training || trainOpened) && <Suspense><TrainDialog root={root} reviewed={reviewed} open={training} onOpenChange={setTraining} /></Suspense>}
    </div>
  );
}

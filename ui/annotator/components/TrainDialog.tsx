import {useEffect, useRef, useState} from 'react';
import {listen} from '@tauri-apps/api/event';
import {CircleStopIcon, PlayIcon, TriangleAlertIcon} from 'lucide-react';
import {CartesianGrid, Line, LineChart, ReferenceLine, XAxis, YAxis} from 'recharts';
import {toast} from 'sonner';
import {errorText} from '@/components/async-button';
import {Alert, AlertDescription, AlertTitle} from '@/components/ui/alert';
import {Badge} from '@/components/ui/badge';
import {Button} from '@/components/ui/button';
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from '@/components/ui/card';
import {ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig} from '@/components/ui/chart';
import {Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle} from '@/components/ui/dialog';
import {Field, FieldDescription, FieldLabel} from '@/components/ui/field';
import {Input} from '@/components/ui/input';
import {Progress} from '@/components/ui/progress';
import {Spinner} from '@/components/ui/spinner';
import {Table, TableBody, TableCell, TableHead, TableHeader, TableRow} from '@/components/ui/table';
import {META, type Label} from '@/labels';
import {desktop, stopTraining, trainModel, type EpochStats, type Progress as Event, type Report} from '../lib/api';

type Start = Extract<Event, {kind: 'start'}>;
type Batch = Extract<Event, {kind: 'batch'}>;

const curves = {
  train: {label: '训练', theme: {light: 'oklch(0.708 0 0)', dark: 'oklch(0.556 0 0)'}},
  val: {label: '验证', theme: {light: 'oklch(0.588 0.158 241.966)', dark: 'oklch(0.746 0.16 232.661)'}},
} satisfies ChartConfig;
/** Epochs without a new validation-loss low, while training loss keeps falling, before we call it overfitting. */
const OVERFIT_PATIENCE = 5;

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const clock = (ms: number) => {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

/** One-click CUDA training on reviewed sessions: data balance, live curves, per-class recall, confusion matrix. */
export function TrainDialog({root, reviewed, open, onOpenChange}: {root: string; reviewed: number; open: boolean; onOpenChange(open: boolean): void}) {
  // An upper bound: training usually ends earlier, once validation stops improving.
  const [epochs, setEpochs] = useState(80);
  const [start, setStart] = useState<Start | null>(null);
  const [batch, setBatch] = useState<Batch | null>(null);
  const [history, setHistory] = useState<EpochStats[]>([]);
  const [report, setReport] = useState<Report | null>(null);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const startedAt = useRef(0);
  const [now, setNow] = useState(0);

  useEffect(() => {
    if (!desktop) return;
    const off = listen<Event>('train-progress', ({payload: e}) => {
      if (e.kind === 'start') { startedAt.current = Date.now(); setStart(e); }
      else if (e.kind === 'batch') setBatch(e);
      else { setBatch(null); setHistory(h => [...h, e]); }
    });
    return () => { void off.then(f => f()); };
  }, []);
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [running]);

  const train = async () => {
    setStart(null); setBatch(null); setHistory([]); setReport(null); setStopping(false); setRunning(true);
    try {
      const r = await trainModel(root, epochs);
      setReport(r);
      toast.success(`训练结束，保存了第 ${r.best_epoch} 轮`);
    } catch (e) { toast.error(errorText(e)); } finally { setRunning(false); setStopping(false); }
  };
  const stop = async () => { setStopping(true); await stopTraining(); };

  const last = history.at(-1);
  const best = history.reduce<EpochStats | null>((b, e) => e.best ? e : b, null);
  const total = start?.epochs ?? epochs;
  const done = batch ? batch.epoch - 1 + batch.batch / batch.batches : last?.epoch ?? 0;
  const elapsed = start && running ? Math.max(now, Date.now()) - startedAt.current : last?.elapsed_ms ?? 0;
  const remaining = done > 0 ? elapsed / done * (total - done) : null;

  return (
    <Dialog open={open} onOpenChange={o => { if (!running) onOpenChange(o); }}>
      <DialogContent className="max-h-[92vh] overflow-y-auto sm:max-w-5xl" showCloseButton={!running}>
        <DialogHeader>
          <DialogTitle>训练模型</DialogTitle>
          <DialogDescription>
            用 {reviewed} 个已校对的会话训练，每个会话最后 20% 留作验证。验证结果最好的一轮存到数据目录的 <code>model</code> 里，一段时间没有进步就自动停。
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-end gap-3">
          <Field className="w-40" data-disabled={running || undefined}>
            <FieldLabel htmlFor="epochs">最多训练轮数</FieldLabel>
            <Input id="epochs" type="number" min={1} max={500} value={epochs} disabled={running} onChange={e => setEpochs(Math.max(1, Math.min(500, Number(e.target.value) || 1)))} />
          </Field>
          {running
            ? <Button variant="destructive" onClick={stop} disabled={stopping}>{stopping ? <Spinner data-icon="inline-start" /> : <CircleStopIcon data-icon="inline-start" />}{stopping ? '正在停止…' : '停止并保留最佳'}</Button>
            : <Button onClick={train} disabled={!desktop || reviewed === 0}><PlayIcon data-icon="inline-start" />{report ? '重新训练' : '开始训练'}</Button>}
          {running && <FieldDescription className="pb-2">停下来也会保留目前最好的一轮。</FieldDescription>}
        </div>

        {running && (
          <div className="flex flex-col gap-2" data-testid="train-progress">
            <Progress value={done / total * 100} />
            <div className="flex flex-wrap justify-between gap-x-4 font-mono text-xs text-muted-foreground">
              <span>{start ? `第 ${Math.min(total, Math.floor(done) + 1)}/${total} 轮${batch ? ` · 批次 ${batch.batch}/${batch.batches} · 本轮损失 ${batch.loss.toFixed(3)}` : ''}` : '准备数据、初始化 CUDA…'}</span>
              <span>已用 {clock(elapsed)}{remaining !== null && ` · 剩余约 ${clock(remaining)}`}{last && ` · 学习率 ${last.lr.toExponential(1)}`}</span>
            </div>
          </div>
        )}

        {start && <DataBalance start={start} />}
        {history.length > 0 && <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat title="最佳验证准确率" value={best ? pct(best.val_balanced) : '—'} note={best ? `第 ${best.epoch} 轮，已保存` : ''} />
            <Stat title="本轮验证准确率" value={pct(last!.val_balanced)}
              note={last!.stale > 0 ? `已 ${last!.stale} 轮没有提升，满 ${last!.patience} 轮自动停止` : `训练集 ${pct(last!.train_balanced)}`} />
            <Stat title="训练损失" value={last!.train_loss.toFixed(3)} note={trend(history.map(h => h.train_loss))} />
            <Stat title="验证损失" value={last!.val_loss.toFixed(3)} note={trend(history.map(h => h.val_loss))} />
          </div>
          <Overfitting history={history} />
          <div className="grid gap-3 md:grid-cols-2">
            <Curve title="损失" description="训练损失带了标签平滑和类别加权，比验证损失高是正常的" history={history} best={best?.epoch} keys={['train_loss', 'val_loss']} />
            <Curve title="准确率" description="各类别分别算，再取平均；竖线是保存的那一轮" history={history} best={best?.epoch} keys={['train_balanced', 'val_balanced']} percent />
          </div>
          <Recall labels={start?.labels ?? []} stats={last!} best={best} />
        </>}

        {report && <Result report={report} />}
      </DialogContent>
    </Dialog>
  );
}

function Stat({title, value, note}: {title: string; value: string; note: string}) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardDescription>{title}</CardDescription>
        <CardTitle className="font-mono text-xl tabular-nums">{value}</CardTitle>
      </CardHeader>
      <CardContent className="text-xs text-muted-foreground">{note || '\u00a0'}</CardContent>
    </Card>
  );
}

/** Change over the last few epochs, in words. */
function trend(values: number[]) {
  if (values.length < 2) return '';
  const recent = values.slice(-3), d = recent.at(-1)! - recent[0];
  return Math.abs(d) < 0.005 ? '基本持平' : d < 0 ? `近 ${recent.length} 轮下降 ${(-d).toFixed(3)}` : `近 ${recent.length} 轮上升 ${d.toFixed(3)}`;
}

function DataBalance({start}: {start: Start}) {
  const max = Math.max(1, ...start.train_counts);
  const missingTrain = start.labels.filter((_, i) => start.train_counts[i] === 0);
  const missingVal = start.labels.filter((_, i) => start.train_counts[i] > 0 && start.val_counts[i] === 0);
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>数据分布</CardTitle>
        <CardDescription>按帧计。</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="grid grid-cols-[4rem_minmax(0,1fr)_5rem_5rem] items-center gap-x-3 gap-y-1.5 text-sm">
          <span /><span /><span className="text-right text-xs text-muted-foreground">训练</span><span className="text-right text-xs text-muted-foreground">验证</span>
          {start.labels.map((l, i) => <BalanceRow key={l} label={l} train={start.train_counts[i]} val={start.val_counts[i]} max={max} />)}
        </div>
        {(missingTrain.length > 0 || missingVal.length > 0) && (
          <Alert>
            <TriangleAlertIcon />
            <AlertTitle>部分类别数据不足</AlertTitle>
            <AlertDescription>
              {missingTrain.length > 0 && <p>{missingTrain.map(l => META[l].name).join('、')}没有训练数据，模型学不会，先去采集。</p>}
              {missingVal.length > 0 && <p>{missingVal.map(l => META[l].name).join('、')}没有验证数据，没法评估。会话录长一点，最后 20% 才是验证用的。</p>}
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}

function BalanceRow({label, train, val, max}: {label: Label; train: number; val: number; max: number}) {
  return <>
    <span className="flex items-center gap-1.5"><span className="size-2 shrink-0 rounded-full" style={{background: META[label].color}} />{META[label].name}</span>
    <div className="h-2 rounded-full bg-muted"><div className="h-full rounded-full" style={{width: `${train / max * 100}%`, background: META[label].color}} /></div>
    <span className="text-right font-mono text-xs tabular-nums">{train}</span>
    <span className="text-right font-mono text-xs tabular-nums">{val}</span>
  </>;
}

function Overfitting({history}: {history: EpochStats[]}) {
  const low = history.reduce((b, e) => e.val_loss < b.val_loss ? e : b);
  const last = history.at(-1)!;
  if (last.epoch - low.epoch < OVERFIT_PATIENCE || last.train_loss >= history[history.findIndex(e => e === low)].train_loss) return null;
  return (
    <Alert>
      <TriangleAlertIcon />
      <AlertTitle>可能在过拟合</AlertTitle>
      <AlertDescription>验证损失第 {low.epoch} 轮之后就没再降，训练损失却还在降。接着练用处不大，补些不一样的数据更管用。</AlertDescription>
    </Alert>
  );
}

function Curve({title, description, history, best, keys, percent}: {title: string; description: string; history: EpochStats[]; best?: number; keys: [keyof EpochStats, keyof EpochStats]; percent?: boolean}) {
  const data = history.map(h => ({epoch: h.epoch, train: h[keys[0]] as number * (percent ? 100 : 1), val: h[keys[1]] as number * (percent ? 100 : 1)}));
  const format = (v: number) => percent ? `${v.toFixed(1)}%` : v.toFixed(3);
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={curves} className="aspect-auto h-52 w-full" data-testid={`curve-${keys[1]}`}>
          <LineChart data={data} margin={{left: 0, right: 8, top: 8}}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="epoch" type="number" domain={[1, 'dataMax']} allowDecimals={false} tickLine={false} axisLine={false} />
            <YAxis width={44} tickLine={false} axisLine={false} domain={percent ? [0, 100] : [0, 'auto']} tickFormatter={(v: number) => percent ? `${v}%` : v.toFixed(2)} />
            {best !== undefined && <ReferenceLine x={best} stroke="var(--color-val)" strokeDasharray="4 4" />}
            <ChartTooltip content={<ChartTooltipContent labelFormatter={(_, p) => `第 ${p?.[0]?.payload?.epoch} 轮`} formatter={(v, name) => (
              <div className="flex w-full items-center gap-2">
                <span className="size-2.5 rounded-[2px]" style={{background: `var(--color-${String(name)})`}} />
                <span className="text-muted-foreground">{curves[name as keyof typeof curves].label}</span>
                <span className="ml-auto font-mono tabular-nums">{format(Number(v))}</span>
              </div>
            )} />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Line dataKey="train" type="monotone" stroke="var(--color-train)" strokeWidth={2} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
            <Line dataKey="val" type="monotone" stroke="var(--color-val)" strokeWidth={2} dot={data.length < 25} isAnimationActive={false} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

function Recall({labels, stats, best}: {labels: Label[]; stats: EpochStats; best: EpochStats | null}) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>各类别验证召回率</CardTitle>
        <CardDescription>第 {stats.epoch} 轮，每类验证帧里认对的比例。竖线是保存的那一轮。</CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-[4rem_minmax(0,1fr)_4rem] items-center gap-x-3 gap-y-2 text-sm" data-testid="recall">
        {labels.map((l, i) => {
          const r = stats.class_recall[i], b = best?.class_recall[i];
          return <div key={l} className="contents">
            <span className="flex items-center gap-1.5"><span className="size-2 shrink-0 rounded-full" style={{background: META[l].color}} />{META[l].name}</span>
            <div className="relative h-2.5 rounded-full bg-muted">
              {r != null && <div className="h-full rounded-full transition-[width]" style={{width: `${r * 100}%`, background: META[l].color}} />}
              {b != null && <div className="absolute -inset-y-1 w-0.5 rounded-full bg-foreground" style={{left: `calc(${b * 100}% - 1px)`}} />}
            </div>
            <span className="text-right font-mono text-xs tabular-nums">{r == null ? <Badge variant="outline">无样本</Badge> : pct(r)}</span>
          </div>;
        })}
      </CardContent>
    </Card>
  );
}

function Result({report}: {report: Report}) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {report.stopped ? '已停止' : report.early_stopped ? '已自动停止' : '训练完成'}<Badge variant="secondary">第 {report.best_epoch} 轮 · {pct(report.val_balanced)}</Badge>
        </CardTitle>
        <CardDescription>
          {report.early_stopped && '验证准确率好几轮没进步，提前停了。'}
          第 {report.best_epoch} 轮存到了 <code>model</code>（训练 {report.train_samples} 帧，验证 {report.val_samples} 帧），去 Gesture Pad 里加载它。下表每行是真实类别，格子是被认成各类的比例。
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table data-testid="confusion" className="table-fixed">
          <TableHeader>
            <TableRow>
              <TableHead className="w-24">真实＼预测</TableHead>
              {report.labels.map(l => <TableHead key={l} className="text-center">{META[l].name}</TableHead>)}
              <TableHead className="text-right">样本</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {report.confusion.map((row, i) => {
              const total = row.reduce((a, b) => a + b, 0);
              return (
                <TableRow key={i}>
                  <TableHead>{META[report.labels[i]].name}</TableHead>
                  {row.map((n, j) => {
                    const share = total ? n / total : 0;
                    return (
                      <TableCell key={j} className="text-center font-mono tabular-nums" title={`${n} 帧`}
                        style={{background: n ? `color-mix(in oklab, ${i === j ? 'oklch(0.723 0.219 149.579)' : 'oklch(0.637 0.237 25.331)'} ${Math.round(15 + share * 55)}%, transparent)` : undefined}}>
                        {total ? `${Math.round(share * 100)}%` : '—'}
                      </TableCell>
                    );
                  })}
                  <TableCell className="text-right font-mono text-muted-foreground tabular-nums">{total}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

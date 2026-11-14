import {useState, type ComponentProps} from 'react';
import {toast} from 'sonner';
import {Button} from '@/components/ui/button';
import {Spinner} from '@/components/ui/spinner';

export const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);

/** Button for async actions: disabled with a spinner while running, errors shown as a toast. */
export function AsyncButton({onClick, disabled, children, ...props}: Omit<ComponentProps<typeof Button>, 'onClick'> & {onClick: () => unknown}) {
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    try { await onClick(); } catch (e) { toast.error(errorText(e)); } finally { setBusy(false); }
  };
  return (
    <Button disabled={disabled || busy} onClick={run} {...props}>
      {busy && <Spinner data-icon="inline-start" />}
      {children}
    </Button>
  );
}

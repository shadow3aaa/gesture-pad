import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import '@/globals.css';
import {ThemeProvider} from '@/theme';
import {Toaster} from '@/components/ui/sonner';
import App from './App';
import {Overlay} from './Overlay';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {location.hash === '#overlay' ? <Overlay /> : (
      <ThemeProvider>
        <App />
        <Toaster position="bottom-center" />
      </ThemeProvider>
    )}
  </StrictMode>,
);

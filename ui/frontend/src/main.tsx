import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource/martian-mono/400.css'
import '@fontsource/martian-mono/700.css'
import '@fontsource/instrument-sans/400.css'
import '@fontsource/instrument-sans/500.css'
import '@fontsource/instrument-sans/600.css'
import './styles.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

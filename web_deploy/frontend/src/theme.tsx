// Dark or light, like the desktop app's toggle (frontend/onyx_desktop/ui/appearance.py). Dark unless chosen
// otherwise; the choice is this browser's own. index.html applies it before the first paint.

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type ThemeMode = "dark" | "light";

const KEY = "onyx.theme";

function remembered(): ThemeMode {
  try {
    return localStorage.getItem(KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

interface Theme {
  mode: ThemeMode;
  toggle: () => void;
}

const ThemeContext = createContext<Theme>({ mode: "dark", toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>(remembered);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
    try {
      localStorage.setItem(KEY, mode);
    } catch {
      /* private window: the choice lasts for this page only */
    }
  }, [mode]);

  const value = useMemo(() => ({ mode, toggle: () => setMode((m) => (m === "dark" ? "light" : "dark")) }), [mode]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);

import { useEffect, useState } from "react";

// Small, defensive localStorage-backed state. Reads/writes are wrapped so the
// app still works in private windows or when storage is blocked.
export function useLocalStorage(key: string, initial: string): [string, (v: string) => void] {
  const [value, setValue] = useState<string>(() => {
    try {
      return window.localStorage.getItem(key) ?? initial;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      /* storage unavailable — keep in-memory only */
    }
  }, [key, value]);

  return [value, setValue];
}

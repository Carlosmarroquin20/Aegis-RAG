/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Aegis palette: deep slate ground with a shielded-blue accent.
        aegis: {
          bg: "#0b1220",
          panel: "#111a2e",
          border: "#1f2c47",
          accent: "#3b82f6",
          accentsoft: "#1d4ed8",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

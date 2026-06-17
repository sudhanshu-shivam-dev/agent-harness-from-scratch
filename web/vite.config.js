import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// `base: "./"` makes the build work from any static host or subpath
// (GitHub Pages, Netlify, Vercel, or imported into Lovable).
export default defineConfig({
    plugins: [react()],
    base: "./",
});

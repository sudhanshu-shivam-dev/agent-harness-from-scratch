# ReAct Agent Playground (web demo)

An interactive, browser-only demo of
[agent-harness-from-scratch](../). It runs a deterministic TypeScript mock of the
Python ReAct agent — the same think → act → observe loop, tool abstraction, and
run stats — with **no backend and no API key**.

## Develop

```bash
cd web
npm install
npm run dev      # http://localhost:5173
```

## Build

```bash
npm run build    # outputs static files to web/dist/
npm run preview  # serve the production build locally
```

## Deploy

`web/dist/` is a static bundle — host it anywhere:

- **Vercel / Netlify**: set the project root to `web/`, build command
  `npm run build`, output dir `dist`.
- **GitHub Pages**: publish `web/dist/` (the Vite `base: "./"` makes it work from
  any subpath).

Then paste the live URL into the root `README.md` under **Live demo**.

## What it demonstrates

- The `think() → act() → observe()` loop, with each step rendered as it happens.
- A typed tool abstraction with auto-generated JSON schemas (calculator,
  web-search stub, datetime).
- Run stats: steps, estimated tokens, stop reason, success badge.

import { useEffect, useRef, useState } from "react";
import { EXAMPLES, runAgent, TOOLS, type RunResult, type Step } from "./agent";

const REPO_URL = "https://github.com/sudhanshu-shivam-dev/agent-harness-from-scratch";

export default function App() {
  const [prompt, setPrompt] = useState("What is 23 times 17?");
  const [result, setResult] = useState<RunResult | null>(null);
  const [visibleSteps, setVisibleSteps] = useState(0);
  const [running, setRunning] = useState(false);
  const [showSchema, setShowSchema] = useState(false);
  const timers = useRef<number[]>([]);

  // Reveal steps one-by-one for a "watch it think" effect.
  useEffect(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    if (!result) return;
    setVisibleSteps(0);
    setRunning(true);
    result.steps.forEach((_, i) => {
      const id = window.setTimeout(() => {
        setVisibleSteps(i + 1);
        if (i === result.steps.length - 1) setRunning(false);
      }, 450 * (i + 1));
      timers.current.push(id);
    });
    return () => timers.current.forEach(clearTimeout);
  }, [result]);

  const run = (text: string) => {
    const t = text.trim();
    if (!t) return;
    setPrompt(t);
    setResult(runAgent(t));
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>
            ReAct Agent <span className="accent">Playground</span>
          </h1>
          <p className="subtitle">think → act → observe, built from scratch</p>
        </div>
        <a className="ghbtn" href={REPO_URL} target="_blank" rel="noreferrer">
          ★ View on GitHub
        </a>
      </header>

      <section className="prompt-row">
        <input
          className="prompt-input"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run(prompt)}
          placeholder="Ask the agent something…"
        />
        <button className="run-btn" onClick={() => run(prompt)} disabled={running}>
          {running ? "Running…" : "Run ▸"}
        </button>
      </section>

      <div className="chips">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => run(ex)}>
            {ex}
          </button>
        ))}
      </div>

      <div className="grid">
        <main className="trajectory">
          <h2>Trajectory</h2>
          {!result && <p className="empty">Run a prompt to watch the agent reason.</p>}
          {result &&
            result.steps.slice(0, visibleSteps).map((s) => <StepCard key={s.index} step={s} />)}
          {result && visibleSteps >= result.steps.length && (
            <div className="answer">
              <span className="answer-label">ANSWER</span>
              {result.answer}
            </div>
          )}
        </main>

        <aside className="sidebar">
          <div className="stats card">
            <h2>Run stats</h2>
            <Stat label="Steps" value={result ? String(result.steps.length) : "—"} />
            <Stat label="Tokens (est.)" value={result ? String(result.tokens) : "—"} />
            <Stat label="Stop reason" value={result ? result.stopReason : "—"} />
            <div className="stat">
              <span>Status</span>
              {result ? (
                <span className={`badge ${result.success ? "ok" : "fail"}`}>
                  {result.success ? "success" : "stopped"}
                </span>
              ) : (
                <span>—</span>
              )}
            </div>
          </div>

          <div className="card tools">
            <h2>Tools</h2>
            {TOOLS.map((t) => (
              <div key={t.name} className="tool">
                <code className="tool-name">{t.name}</code>
                <span className="tool-desc">{t.description}</span>
              </div>
            ))}
            <button className="link" onClick={() => setShowSchema((v) => !v)}>
              {showSchema ? "Hide" : "Show"} auto-generated JSON schema
            </button>
            {showSchema && (
              <pre className="schema">{JSON.stringify(TOOLS, null, 2)}</pre>
            )}
          </div>
        </aside>
      </div>

      <footer className="footer">
        Deterministic in-browser mock of{" "}
        <a href={REPO_URL} target="_blank" rel="noreferrer">
          agent-harness-from-scratch
        </a>{" "}
        — no API key, no backend.
      </footer>
    </div>
  );
}

function StepCard({ step }: { step: Step }) {
  if (!step.action) return null; // final-answer step is rendered separately
  return (
    <div className="step">
      <div className="step-head">
        <span className="step-idx">step {step.index}</span>
        <span className="step-thought">{step.thought}</span>
      </div>
      <div className="step-action">
        <span className="tag tag-action">action</span>
        <code>{step.action.name}</code>
        <code className="args">{JSON.stringify(step.action.args)}</code>
      </div>
      <div className="step-obs">
        <span className="tag tag-obs">observation</span>
        <span>{step.observation}</span>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

/**
 * A browser-side mock of the Python ReAct agent in agent-harness-from-scratch.
 *
 * Deterministic, no network, no API key. Mirrors the same think -> act ->
 * observe loop, tool abstraction, and run stats so the playground faithfully
 * demonstrates how the real harness behaves.
 */

export interface ToolSchema {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, { type: string; description?: string }>;
    required: string[];
  };
}

export interface Step {
  index: number;
  thought: string;
  action: { name: string; args: Record<string, unknown> } | null;
  observation: string | null;
}

export interface RunResult {
  answer: string;
  steps: Step[];
  tokens: number;
  stopReason: string;
  success: boolean;
}

/** ~4 chars per token, matching the Python estimator. */
const estimateTokens = (text: string): number =>
  text ? Math.max(1, Math.floor(text.length / 4)) : 0;

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------
const SEARCH_KB: Record<string, string> = {
  "capital of france": "Paris is the capital of France.",
  "capital of japan": "Tokyo is the capital of Japan.",
  "tallest mountain": "Mount Everest is the tallest mountain on Earth at 8,849 m.",
  "speed of light": "The speed of light is approximately 299,792 km/s.",
  "creator of python": "Python was created by Guido van Rossum.",
  python: "Python was created by Guido van Rossum.",
};

/** Safe arithmetic evaluator (shunting-yard -> RPN). No eval(). */
function safeCalc(expression: string): string {
  const tokens = expression.match(/\d+\.?\d*|[+\-*/()%]/g);
  if (!tokens) throw new Error("no expression");
  const prec: Record<string, number> = { "+": 1, "-": 1, "*": 2, "/": 2, "%": 2 };
  const output: string[] = [];
  const ops: string[] = [];
  for (const t of tokens) {
    if (/\d/.test(t)) {
      output.push(t);
    } else if (t in prec) {
      while (ops.length && ops[ops.length - 1] in prec && prec[ops[ops.length - 1]] >= prec[t]) {
        output.push(ops.pop()!);
      }
      ops.push(t);
    } else if (t === "(") {
      ops.push(t);
    } else if (t === ")") {
      while (ops.length && ops[ops.length - 1] !== "(") output.push(ops.pop()!);
      ops.pop();
    }
  }
  while (ops.length) output.push(ops.pop()!);

  const stack: number[] = [];
  for (const t of output) {
    if (/\d/.test(t)) {
      stack.push(parseFloat(t));
    } else {
      const b = stack.pop()!;
      const a = stack.pop()!;
      stack.push(
        t === "+" ? a + b : t === "-" ? a - b : t === "*" ? a * b : t === "%" ? a % b : a / b,
      );
    }
  }
  const result = stack[0];
  return Number.isInteger(result) ? String(result) : String(result);
}

export const TOOLS: ToolSchema[] = [
  {
    name: "calculator",
    description: "Evaluate a basic arithmetic expression and return the result.",
    parameters: {
      type: "object",
      properties: {
        expression: { type: "string", description: "An arithmetic expression, e.g. '23 * 17'." },
      },
      required: ["expression"],
    },
  },
  {
    name: "web_search",
    description: "Look up a fact from a small canned knowledge base (offline stub).",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "The search query." } },
      required: ["query"],
    },
  },
  {
    name: "datetime",
    description: "Return the current date and time in ISO-8601 format.",
    parameters: { type: "object", properties: {}, required: [] },
  },
];

function dispatch(name: string, args: Record<string, unknown>): string {
  if (name === "calculator") {
    try {
      return safeCalc(String(args.expression ?? ""));
    } catch {
      return `Could not evaluate '${args.expression}'.`;
    }
  }
  if (name === "web_search") {
    const q = String(args.query ?? "").toLowerCase();
    for (const [key, value] of Object.entries(SEARCH_KB)) {
      if (q.includes(key)) return value;
    }
    return `No results found for '${args.query}'.`;
  }
  if (name === "datetime") {
    return new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
  }
  return `ERROR: unknown tool '${name}'.`;
}

// ---------------------------------------------------------------------------
// Mock "LLM" decision (same heuristics as the Python MockLLM)
// ---------------------------------------------------------------------------
const WORD_OPS: Record<string, string> = {
  plus: "+",
  add: "+",
  minus: "-",
  subtract: "-",
  times: "*",
  "multiplied by": "*",
  multiply: "*",
  "divided by": "/",
  divide: "/",
};

function extractExpression(text: string): string | null {
  let lowered = text.toLowerCase();
  for (const [word, op] of Object.entries(WORD_OPS)) lowered = lowered.split(word).join(op);
  const candidate = lowered.replace(/[^0-9.+\-*/()% ]/g, " ").replace(/\s+/g, " ").trim();
  if (/\d/.test(candidate) && /[+\-*/%]/.test(candidate)) return candidate;
  return null;
}

function decideTool(text: string): { name: string; args: Record<string, unknown> } | null {
  const lower = text.toLowerCase();
  const expr = extractExpression(text);
  if (expr) return { name: "calculator", args: { expression: expr } };
  if (["time", "date", "day", "today", "now"].some((k) => lower.includes(k)))
    return { name: "datetime", args: {} };
  if (["search", "who", "what is", "capital", "tallest", "speed of"].some((k) => lower.includes(k)))
    return { name: "web_search", args: { query: text } };
  return null;
}

// ---------------------------------------------------------------------------
// The ReAct loop
// ---------------------------------------------------------------------------
export function runAgent(task: string, maxSteps = 6): RunResult {
  const steps: Step[] = [];
  let tokens = estimateTokens(task) + 40; // system prompt + task
  let answer = "";
  let stopReason = "finished";

  const decision = decideTool(task);
  if (decision) {
    // Step 0: think -> tool call
    const observation = dispatch(decision.name, decision.args);
    tokens += 8 + estimateTokens(observation);
    steps.push({
      index: 0,
      thought: `I should use the ${decision.name} tool.`,
      action: decision,
      observation,
    });
    // Step 1: think -> final answer from observation
    answer = `Based on the tool result, the answer is: ${observation}`;
    tokens += estimateTokens(answer);
    steps.push({ index: 1, thought: answer, action: null, observation: "final_answer" });
  } else {
    answer = `I don't have a tool to handle: "${task}".`;
    tokens += estimateTokens(answer);
    steps.push({ index: 0, thought: answer, action: null, observation: "final_answer" });
  }

  if (steps.length > maxSteps) stopReason = "budget: max_steps reached";

  return { answer, steps, tokens, stopReason, success: stopReason === "finished" };
}

export const EXAMPLES = [
  "What is 23 times 17?",
  "Search for the capital of France.",
  "What is today's date?",
  "What is (12 + 8) * 5?",
];

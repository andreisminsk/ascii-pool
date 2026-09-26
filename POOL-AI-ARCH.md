# ASCII Pool v2 — LLM Opponent Architecture

**Date:** 2026-09-26
**Status:** Implemented — `ascii_pool_llm.py`
**Base:** `ascii_pool.py` (v1, heuristic CPU)

## 1. Idea

Same hybrid split as ascii-pong v2: physics does the math, the
LLM makes the judgment call.

- **Physics enumerates** every feasible shot: for each
  (own ball, pocket) pair — ghost-ball contact point, path
  clearance, cut angle, distance → a scored candidate list.
  The code already exists: `best_shot()` in v1.
- **The LLM picks** among the top-N candidates and tweaks power:
  "which ball, which pocket, soft or firm?" — a small,
  few-shot-anchored choice, not ballistic math. Exactly the
  270M-model sweet spot proven in ascii-pong (semi-structured
  `key=value` in, one tiny JSON object out).
- **Fallback ladder:** LLM → v1 heuristic → safety shot.
  Circuit breaker, pre-flight probe, and model picker port
  as-is from `ascii_pong_llm.py`.

## 2. Latency budget

Pool is far more forgiving than pong: the shot happens *after*
the model answers, and "thinking over the table" is natural.
Budget ≈ 2–5 s per shot — even cloud models qualify. The
interesting axis is not latency but **taste**: varying pace,
playing shape for the next ball, taking risky cuts.

## 3. Prompt sketch

```
you=SOLIDS remaining=2,5,7  opp=A,C  eight=8
candidates:
1) ball=5 pocket=top-left  cut=12 dist=18 score=87
2) ball=2 pocket=bot-right cut=41 dist=26 score=61
3) ball=7 pocket=top-mid  cut=63 dist=31 score=34
reply={"pick":1,"power":0.7}
```

## 4. Knobs

| Constant | Default | Meaning |
|---|---|---|
| `LLM_TIMEOUT` | 5 s | per-shot budget |
| `LLM_TOPN` | 5 | candidates offered |
| `LLM_BLEND` | 1.0 | 0 = heuristic pick, 1 = full LLM |
| `LLM_BREAKER` | 3 | failures before fallback |

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Picks illegal targets | clamp to the candidate list only |
| Repetitive play | temperature + contrasting few-shot examples |
| Reasoning models stall | probe detects it, like pong v2 |
| Outage mid-rack | breaker → v1 heuristic, self-heals |

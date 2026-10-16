# Personal Financial Safety Net Agent

A LangGraph multi-agent system that reads a person's insurance policies, loan agreement and finances together,
tests them against life-shock scenarios, and reports coverage gaps with clause-level citations.
Diagnostic only: it never recommends products, and fix-cost estimates are held behind an approval gate.

All documents are **synthetic** (3 personas, 11 documents, 15 labelled cases).

## Architecture

```
Intake -> Scenario Generator -> fan-out per scenario (LangGraph Send)
            [Risk Flagger || Coverage Checker]   independent prompts + independent retrieval queries
            -> deterministic citation check       drops citations whose quote isn't in the retrieved clause
            -> Resolver (larger model)            rules confirmed / dismissed / inconclusive on source text
       -> fan-in -> Metrics (pure Python) -> Writer -> Approval Gate
```

| Piece | File | Notes |
|---|---|---|
| LLM gateway | `llm.py` | Groq -> retries with temperature jitter -> Gemini fallback; Pydantic-validated JSON; semaphore caps concurrency; per-call trace log |
| Schemas | `schema.py` | Typed outputs for every agent |
| Intake / retrieval | `outputs/retrieval.py` | Clause-level chunks with refs like `term_policy.md#5.2`; TF-IDF |
| Agents | `outputs/agents/agents.py` | Scenario Generator, Flagger, Checker, Resolver, Writer, single-agent baseline |
| Citation guard | `outputs/agents/common.py` | Deterministic, unit-tested |
| Money math | `outputs/metrics.py` | Runway, locked-asset access net of penalty, severity. No LLM arithmetic |
| Graph | `outputs/graph/build_graph.py` | Fan-out/fan-in, approval gate |
| Eval | `outputs/eval/run_eval.py` | Multi-agent vs single-agent baseline on a frozen answer key |

Models (Groq free tier): `openai/gpt-oss-20b` for Flagger/Checker/Writer, `openai/gpt-oss-120b` for Resolver and baseline.

## Run

```
pip install -r requirements.txt
# .env: GROQ_API_KEY=..., GEMINI_API_KEY=...
python -m pytest -q outputs/tests          # 6 tests, no API needed (mock LLM)
python main.py persona_c                   # full graph on one persona
python main.py persona_c --approve         # release held fix-cost estimates
python -m outputs.eval.run_eval --runs 3   # eval (needs ~half the free daily Groq quota per run)
```

## Evaluation (13 Oct 2026)

Answer key: 15 cases written and frozen before any agent ran; 7 conflicting-evidence (a rider closes an apparent
gap, or an exclusion/waiting period opens a hidden one), 2 cross-document, 6 simple.
Baseline is deliberately strong: same documents, union of both retrieval sets, the larger model, one pass.

Valid head-to-head case-runs (both systems answered; later runs hit Groq's daily token limit): **43**

| System | Correct | Errors | Tokens / case |
|---|---|---|---|
| Multi-agent (Flagger + Checker + Resolver) | 36/43 (83.7%) | all false positives | ~9,000 |
| Single-agent baseline | 38/43 (88.4%) | all false positives | ~1,900 |

Retrieval recall of hand-labelled key clauses: 100% (documents are short, so this is not the bottleneck).

### What the eval shows
- **Multi-agent did not beat the baseline** on this set; it was slightly worse at ~4.7x the tokens.
- **Every error is a false positive.** Both systems fail A1 in every run (accident; disability rider pays after a
  90-day wait and covers most of the outgo, savings cover the rest). C5 is the same pattern.
- These cases need month-by-month cash-flow arithmetic over waiting periods and partial benefits.
  The design keeps arithmetic out of the LLMs, but no code computes scenario cash flow yet, so the models guess.

### Next steps
1. Add a deterministic scenario cash-flow tool (shortfall = outgo x months - benefits payable - usable assets)
   and pass its output to every agent.
2. Validate on **new** held-out cases (the current 15 have now been seen, so tuning on them would overfit).
3. Re-run with full 3 runs on a fresh daily quota and report ranges.

Raw outputs: `results/eval_summary.json`, `results/eval_traces.json`, `results/eval*.log`.

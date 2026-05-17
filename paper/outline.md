# DiLu-Ollama Paper Outline

## Working Title Candidates

1. Fine-Tuning Local LLM Driving Agents Under Runtime Constraints: A Three-Tier Highway Benchmark with DiLu-Ollama
2. Benchmarking Fine-Tuned Local Language Models for Autonomous Driving in Highway Simulation
3. DiLu-Ollama: Local-First Benchmarking and Fine-Tuning Analysis for LLM Driving Agents

## One-Sentence Contribution

We present a local-first benchmark and analysis pipeline for LLM driving agents and use it to study fine-tuning under realistic runtime constraints, finding promising lightweight gains but sharply reduced benchmark viability at larger model tiers.

## Paper Shape

- Venue style: NeurIPS/ICLR empirical ML paper
- Evidence scope: three-tier highway benchmark only
- Central framing: fine-tuning study measured through a local-first benchmark pipeline
- Claim style: conservative and evidence-bounded

## Section Plan

### Abstract

**Intended takeaway**
- State the problem, method, main findings, and contribution in five compact sentences.

**Allowed claims**
- Local-first LLM driving evaluation is confounded by runtime latency and invalid runs.
- DiLu-Ollama provides a benchmark plus analysis pipeline for studying these effects.
- Lightweight models provide the strongest usable fine-tuning evidence.
- Exact pairwise fine-tuning evidence is currently limited to one ranking-eligible lightweight family.

**Required evidence**
- `analysis/out/publication_three_tier_results_v1/table_1_cross_tier_validity_summary.csv`
- `analysis/out/publication_three_tier_results_v1/table_3_lightweight_exact_pair_deltas.csv`

**Required figures/tables**
- None in-section; refer to Table 1 and Table 3 implicitly.

### 1. Introduction

**Intended takeaway**
- Motivate why local-first LLM driving evaluation is scientifically useful but operationally fragile.

**Allowed claims**
- Runtime latency, timeout collapse, and invalid execution complicate model comparisons in local simulation.
- A benchmark is only useful if it can represent both task quality and benchmark validity.
- Fine-tuning is worth studying in this setting, but claims must be restricted to ranking-eligible pairs.

**Required evidence**
- `analysis/out/publication_three_tier_results_v1/evidence_summary.md`
- `analysis/out/publication_three_tier_results_v1/table_1_cross_tier_validity_summary.csv`

**Required figures/tables**
- Figure 3 in the main paper or early results preview.

**Proposed subsections**
- Problem setting and motivation
- Why local-first evaluation matters
- What this paper studies
- Contributions

### 2. Related Work

**Intended takeaway**
- Position the paper against autonomous-driving LLM papers, PEFT background, and local inference systems work without diluting the main contribution.

**Allowed claims**
- Prior work shows that LLMs can function as planning or decision components in autonomous-driving simulation.
- Prior PEFT work makes local adaptation feasible, and prior systems work makes local runtime constraints scientifically relevant.
- The gap in the literature is not simply better driving behavior, but evaluation that remains trustworthy under local runtime constraints.

**Required evidence**
- `paper/references.bib`
- `paper/related_work_candidates.md`

**Required figures/tables**
- None.

**Proposed subsections**
- LLM driving-agent line
- PEFT and efficient adaptation
- Local runtime constraints and the paper's positioning

### 3. Benchmark and Framework Setup

**Intended takeaway**
- Describe DiLu-Ollama as the measurement scaffold rather than the headline contribution.

**Allowed claims**
- The framework combines local Ollama inference, highway-env simulation, LaMPilot-style task evaluation, and post-hoc study generation.
- The current paper uses the highway benchmark path only.
- Cross-scenario work exists in the framework but is not part of the empirical evidence in this paper.

**Required evidence**
- `README.md` for framing support only
- benchmark implementation already present in the repo
- `analysis/out/publication_three_tier_results_v1/figure_plan.md`

**Required figures/tables**
- Optional framework diagram if needed later; not required for the first Markdown draft.

**Proposed subsections**
- Runtime stack and decision loop
- Benchmark protocol
- Validity handling and timeout policy

### 4. Fine-Tuning Setup

**Intended takeaway**
- Explain how fine-tuned models were produced and how pairwise comparisons are defined.

**Allowed claims**
- The study compares base and fine-tuned Ollama-served models grouped by family.
- Exact fine-tuning claims are limited to ranking-eligible base/fine-tuned pairs.
- Pair coverage is sparse and must be treated as a study limitation, not ignored.

**Required evidence**
- `fine_tuning/README.md`
- `analysis/slm_model_registry.csv`
- `analysis/out/publication_three_tier_results_v1/table_3_lightweight_exact_pair_deltas.csv`
- `analysis/out/publication_three_tier_results_v1/table_4_midclass_screening_summary.csv`

**Required figures/tables**
- Table 3 in main paper
- family coverage note or appendix table if needed

**Proposed subsections**
- Model families and pairing logic
- Fine-tuning pipeline summary
- Pair eligibility rule

### 5. Experimental Design

**Intended takeaway**
- Make the comparison protocol reproducible and explain why validity is part of the outcome.

**Allowed claims**
- Models are evaluated under a shared highway benchmark protocol.
- `driving_score_v2` is the headline task metric, but interpretation is conditioned on benchmark validity.
- Invalid runs are excluded from ranking but retained for screening and failure analysis.

**Required evidence**
- `analysis/out/publication_three_tier_results_v1/table_1_cross_tier_validity_summary.csv`
- `analysis/out/publication_three_tier_results_v1/figure_plan.md`

**Required figures/tables**
- Table 1 in main paper
- Figure 3 in main paper

**Proposed subsections**
- Tier definitions
- Metrics and validity rules
- Main comparison questions

### 6. Results

**Intended takeaway**
- Separate what is established from what is only screening evidence or failure evidence.

**Allowed claims**
- Lightweight is the only results-ready tier.
- The best valid lightweight model is Llama 3.2 3B Base.
- The only exact eligible fine-tuning gain is the Phi lightweight pair.
- Midclass remains screening-only.
- Highclass results are failure-analysis evidence about runtime scalability, not competitive task ranking.

**Required evidence**
- `analysis/out/publication_three_tier_results_v1/table_1_cross_tier_validity_summary.csv`
- `analysis/out/publication_three_tier_results_v1/table_2_lightweight_leaderboard.csv`
- `analysis/out/publication_three_tier_results_v1/table_3_lightweight_exact_pair_deltas.csv`
- `analysis/out/publication_three_tier_results_v1/table_4_midclass_screening_summary.csv`
- `analysis/out/publication_three_tier_results_v1/table_5_highclass_failure_summary.csv`

**Required figures/tables**
- Main paper:
  - Table 1
  - Table 2
  - Table 3
  - Figure 3
  - Figure 4
- Appendix candidates:
  - Figure 1
  - Figure 2
  - Figure 5
  - Figure 6
  - Figure 7
  - Table 4
  - Table 5

**Proposed subsections**
- 5.1 Tier-level screening overview
- 5.2 Lightweight comparative evidence
- 5.3 Exact fine-tuning evidence in lightweight
- 5.4 Midclass screening outcomes
- 5.5 Highclass runtime-collapse analysis

### 6. Limitations

**Intended takeaway**
- Make the evidence boundaries explicit and preempt reviewer over-interpretation.

**Allowed claims**
- Exact pair coverage is sparse.
- Midclass and highclass do not support broad family-level fine-tuning conclusions.
- The paper does not yet establish cross-scenario generalization.
- The current timeout policy is part of the local runtime regime studied here.

**Required evidence**
- `analysis/out/publication_three_tier_results_v1/evidence_summary.md`
- `analysis/out/publication_three_tier_results_v1/table_1_cross_tier_validity_summary.csv`

**Required figures/tables**
- None mandatory.

### 7. Conclusion

**Intended takeaway**
- End with a tight synthesis: the framework is useful because it exposes both gains and failure modes.

**Allowed claims**
- DiLu-Ollama enables reproducible local-first evaluation and manuscript-facing analysis.
- Fine-tuning is promising in lightweight settings but remains underdetermined beyond them.
- Practical benchmark viability currently declines with tier under the tested local runtime policy.

**Required evidence**
- `analysis/out/publication_three_tier_results_v1/evidence_summary.md`

**Required figures/tables**
- None mandatory.

## Figure and Table Placement

### Main paper

- Table 1: cross-tier validity summary
- Table 2: lightweight leaderboard
- Table 3: lightweight exact-pair deltas
- Figure 3: cross-tier validity bar chart
- Figure 4: cross-tier Pareto plot

### Appendix / supplement

- Figure 1: lightweight task-first compare plot
- Figure 2: lightweight efficiency companion plot
- Figure 5: lightweight Pareto plot
- Figure 6: midclass Pareto plot
- Figure 7: highclass Pareto plot
- Table 4: midclass screening summary
- Table 5: highclass failure summary

## Explicit Non-Claims

- Do not claim a general fine-tuning win across all model families.
- Do not claim larger models outperform smaller ones.
- Do not claim cross-scenario empirical generalization in this first paper.
- Do not claim the benchmark is universally stable across scales.

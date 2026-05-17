# Results Analysis Memo for Paper Writing

This memo summarizes the strict evidence that is safe to carry into the paper narrative.

## Evidence-safe findings

1. Cross-tier validity is the first result, not a side note.
- Lightweight: `7/14` valid
- Midclass: `3/6` valid
- Highclass: `0/4` valid

2. Lightweight is the only evidence-ready tier.
- This is the only tier with enough valid rows for both ranking and one exact eligible pairwise fine-tuning claim.

3. The only exact eligible fine-tuning pair is the Phi lightweight family.
- `delta_driving_score_v2 = +0.0419`
- `delta_overall_score_mean = +0.1006`
- `delta_decision_latency_ms_avg_mean = -106.264975`

4. Midclass is analytically useful but still screening-only.
- A best valid model exists in each family ranking summary.
- No exact family pair is fully ranking-eligible.

5. Highclass is a runtime-scalability finding.
- All four 14B models are invalid due to timeout-collapse / incomplete execution.
- Do not treat this tier as a leaderboard.

## Main-paper assets

- Table 1: cross-tier validity summary
- Table 2: lightweight leaderboard
- Table 3: exact lightweight pair delta
- Figure 3: cross-tier validity figure
- Figure 4: cross-tier Pareto figure

## Appendix assets

- Table 4: midclass screening summary
- Table 5: highclass failure summary
- Figure 1 / 2 / 5 / 6 / 7 as supporting visuals

## Unsafe claims

- “Fine-tuning improves performance across model families.”
- “Larger local models perform better than smaller ones.”
- “The paper establishes cross-scenario generalization.”
- “The current benchmark is robust across all model scales.”

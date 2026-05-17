# Figure and Table Inventory

This inventory maps existing bundle assets to manuscript roles. It is intentionally conservative: only assets that directly support the current highway-only fine-tuning narrative should appear in the main paper.

| Asset | Role in paper | Recommended placement | Source artifact | Current status | Notes |
| --- | --- | --- | --- | --- | --- |
| Figure 1: local evaluation and validity pipeline | Explains the highway-only measurement scaffold and why invalid runs are retained as outcomes | Main paper | `paper/latex/figures/local_eval_validity_pipeline.png` | ready | New conceptual schematic generated with `scientific-schematics` |
| Figure 2: benchmark-test specification | Explains the fixed highway benchmark suite, task taxonomy, and validity-aware evaluation path before the quantitative results | Main paper | `paper/latex/figures/benchmark_test_specification.png` | ready | New conceptual schematic generated with `scientific-schematics` |
| Table 1: cross-tier validity summary | Establish tier-level screening status and show why later claims are constrained | Main paper | `analysis/out/publication_three_tier_results_v1/table_1_cross_tier_validity_summary.csv` | ready | Use early in Results or Experimental Design |
| Table 2: lightweight leaderboard | Main comparative evidence for the evidence-ready tier | Main paper | `analysis/out/publication_three_tier_results_v1/table_2_lightweight_leaderboard.csv` | ready | Best place to anchor lightweight narrative |
| Table 3: lightweight exact pair deltas | Main fine-tuning evidence table | Main paper | `analysis/out/publication_three_tier_results_v1/table_3_lightweight_exact_pair_deltas.csv` | ready | Only exact eligible pair should be shown |
| Table 4: midclass screening summary | Supports screening-only interpretation and blocked pairwise claims | Appendix | `analysis/out/publication_three_tier_results_v1/table_4_midclass_screening_summary.csv` | ready | Cite in text when discussing midclass limitations |
| Table 5: highclass failure summary | Documents timeout-collapse and invalid highclass runs | Appendix | `analysis/out/publication_three_tier_results_v1/table_5_highclass_failure_summary.csv` | ready | Use as failure-context evidence, not ranking evidence |
| Figure 3: cross-tier validity bar chart | Best visual for the central “validity decreases with tier” message | Main paper | `analysis/out/publication_three_tier_results_v1/figures/figure_3_cross_tier_validity.png` | ready | Strong early figure |
| Figure 4: cross-tier Pareto plot | Shows latency-quality frontier without collapsing metrics | Main paper | `analysis/out/publication_three_tier_results_v1/figures/figure_4_cross_tier_pareto.png` | ready | Best companion to Table 2 and Table 3 |
| Figure 1: lightweight task-first compare plot | Supports within-tier ranking detail | Appendix or supplement | `analysis/out/publication_three_tier_results_v1/figures/figure_1_lightweight_task_summary.png` | ready | Use if reviewer wants a denser view |
| Figure 2: lightweight efficiency companion plot | Adds throughput and latency detail for lightweight | Appendix or supplement | `analysis/out/publication_three_tier_results_v1/figures/figure_2_lightweight_efficiency.png` | ready | Useful if main paper needs efficiency context |
| Figure 5: lightweight Pareto plot | Within-tier Pareto frontier for lightweight | Appendix | `analysis/out/publication_three_tier_results_v1/figures/figure_5_lightweight_pareto.png` | ready | Nice supplement, not essential in main paper |
| Figure 6: midclass Pareto plot | Visual support for provisional midclass screening | Appendix | `analysis/out/publication_three_tier_results_v1/figures/figure_6_midclass_pareto.png` | ready | Must remain explicitly provisional |
| Figure 7: highclass Pareto plot | Failure-context plot for highclass | Appendix | `analysis/out/publication_three_tier_results_v1/figures/figure_7_highclass_pareto.png` | ready | Do not treat as a valid competitive frontier |

## Main-Paper Figure Order

1. Figure 1: local evaluation and validity pipeline
2. Figure 2: benchmark-test specification
3. Figure 3: cross-tier validity bar chart
4. Table 2: lightweight leaderboard
5. Table 3: lightweight exact pair deltas
6. Figure 4: cross-tier Pareto plot
7. Table 1: cross-tier validity summary

## Appendix Order

1. Figure 1: lightweight task-first compare plot
2. Figure 2: lightweight efficiency companion plot
3. Figure 5: lightweight Pareto plot
4. Table 4: midclass screening summary
5. Figure 6: midclass Pareto plot
6. Table 5: highclass failure summary
7. Figure 7: highclass Pareto plot

## Caption Intent

- Figure 1 should say explicitly that it is a **measurement pipeline** and includes validity handling.
- Figure 2 should say explicitly that it is a **benchmark-test specification** and not a performance summary.
- Figure 3 should say explicitly that it shows **ranking eligibility**, not task performance.
- Figure 4 should say explicitly that it includes **ranking-eligible models only**.
- Table 3 should say explicitly that it shows **the only exact eligible lightweight pair**.
- Table 4 and Table 5 should be labeled as **screening/failure context**, not main performance tables.

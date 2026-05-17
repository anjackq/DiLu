# Self-Review: Draft 1

Review basis:
- `paper/draft.md`
- `paper/latex/main.tex`
- `paper/latex/sections/*.tex`

## Structure review

- The draft still contains all required core sections for a first ML paper draft.
- The strongest sections are now the Abstract, Introduction, and Results, which correctly carry the paper’s main argument.
- The Results section is still ordered correctly from screening overview to tier-specific evidence.
- The Limitations section remains explicit and non-defensive, which is good for reviewer trust.
- The LaTeX manuscript now functions as the canonical paper workspace rather than a direct export target.

## Logic consistency review

- The revised draft consistently treats benchmark validity as part of the empirical outcome rather than as a filtering detail.
- The stronger conference framing is still evidence-safe because exact fine-tuning claims remain restricted to the Phi lightweight pair.
- Midclass is now presented more clearly as partial viability without clean causal comparison.
- Highclass is now framed more effectively as a meaningful failure-analysis result rather than as “missing performance.”

## Citation review

- The draft now uses citation keys that are present in `paper/references.bib`.
- The verified set is now strong enough to support a compact Related Work section without placeholders.
- The literature coverage is still intentionally narrow, but it now spans the three lanes the paper actually needs: autonomous-driving LLMs, PEFT, and local inference constraints.
- `liang2022helm` is a justified addition because it supports the multidimensional-evaluation framing without broadening the claim package.
- `highway-env` should remain on the official project citation unless the upstream project later adopts a different canonical citation.

## Figure/table review

- The draft now includes explicit callouts to the main figures and tables:
  - `Table~\\ref{tab:validity-summary}`
  - `Table~\\ref{tab:lightweight-leaderboard}`
  - `Table~\\ref{tab:lightweight-pair-delta}`
  - `Figure~\\ref{fig:cross-tier-validity}`
  - `Figure~\\ref{fig:cross-tier-pareto}`
- Midclass and highclass appendix-table callouts are also in place.
- The next step is not to add more callouts, but to make sure the eventual LaTeX manuscript uses these exact labels consistently.

## Writing review

- The draft now reads more like a conference paper and less like repo documentation.
- The stronger framing around runtime validity and benchmark breakdown is a real improvement.
- The new Related Work section is compact and disciplined; it positions the paper without diluting the main measurement argument.
- The setup sections are now leaner and more conference-appropriate; they explain the benchmark without absorbing too much space before Results.
- The Abstract, first two Introduction paragraphs, Results openings, and Conclusion now read more naturally and less formulaically.
- `writing-anti-ai` should wait until after one more content pass and before LaTeX polishing.

## Main revision tasks

1. Tighten section-to-figure placement once page limits are visible in LaTeX.
2. Run a final prose-polish pass before external circulation.
3. Decide whether the review copy should stay in `preprint` mode or return to anonymous submission mode.

## Review verdict

- Draft 1 has a credible conference-paper narrative.
- The strongest contribution is now clear: fine-tuning claims are only meaningful inside the region where benchmark validity survives local runtime constraints.
- Safe for reviewer-facing internal circulation in the current LaTeX form.

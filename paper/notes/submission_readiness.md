# Submission Readiness Note

Review basis:
- `paper/latex/main.tex`
- `paper/latex/sections/*.tex`
- `paper/latex/references.bib`

## Current readiness

- The manuscript now has a coherent conference-paper narrative in LaTeX.
- The strongest through-line is stable across Abstract, Introduction, Results, Limitations, and Conclusion:
  - benchmark validity is part of the empirical outcome
  - lightweight is the only evidence-ready tier
  - the Phi pair is the only exact eligible fine-tuning comparison
  - highclass is a failure-analysis result rather than missing evidence
- Figure and table captions are clearer about interpretation boundaries.

## Citation decisions

- Added `liang2022helm` as a targeted evaluation-framework citation.
- Retained `leurent2018highwayenv` as the manuscript citation for `highway-env`.
  - Reason: the official `highway-env` documentation and upstream README still point to the GitHub-style project citation block rather than to a more formal canonical paper citation.
- No broad literature expansion was introduced in this pass.

## Remaining issues before external circulation

- Recompile after each substantive edit because table and figure placement can still shift.
- Minor LaTeX underfull box warnings remain, but there are no build-blocking errors and no unresolved references.
- The argument is intentionally conservative. Any stronger novelty pitch should come from framing polish, not broader empirical claims.

## Recommended next pass

1. tighten page-level figure and table placement in LaTeX
2. run one final prose-polish pass
3. decide whether to keep `preprint` mode for internal review or switch back to anonymous submission mode

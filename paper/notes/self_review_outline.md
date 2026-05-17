# Self-Review: Outline

Review basis: `paper/outline.md`

## Structure review

- Abstract, Introduction, Framework Setup, Fine-Tuning Setup, Experimental Design, Results, Limitations, and Conclusion are all present.
- The Results section is correctly split into lightweight evidence, exact pairwise evidence, midclass screening, and highclass failure analysis.
- Main-paper versus appendix figure roles are explicit.

## Logic consistency review

- The paper framing is internally consistent: it is a fine-tuning study constrained by benchmark validity.
- The outline does not overclaim beyond the current evidence bundle.
- Cross-scenario support is treated as framework scope rather than empirical evidence, which is correct for the first paper.

## Figure/table review

- Table 1 and Figure 3 both support the “validity decreases with tier” message.
- Table 3 is correctly elevated because it is the only exact eligible fine-tuning evidence.
- Highclass assets are appropriately demoted to appendix/failure context.

## Main issues to keep watching

1. The framework section could become too large relative to the evidence.
- Keep it concise and use it to support the empirical story rather than turning the paper into a systems/framework paper.

2. The fine-tuning story is real but narrow.
- Preserve the exact-pair rule everywhere in prose.

3. The introduction must not imply that multiple exact fine-tuning wins already exist.
- The contribution list should say “initial evidence” rather than “broad improvement.”

## Review verdict

- Outline is structurally sound for a conservative first paper.
- Safe to use as the basis for a fuller Markdown draft and then LaTeX migration.

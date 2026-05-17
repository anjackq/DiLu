# LaTeX Workspace

This directory is the IEEE journal-style LaTeX workspace for the DiLu-Ollama paper.

## Style

- Target venue: `IEEE Transactions on Intelligent Vehicles (T-IV)`
- Current document class: `IEEEtran` journal mode
- Manuscript type: regular paper
- The workspace is structured as a self-contained IEEE journal manuscript rather than a conference preprint template.

## Structure

- [main.tex](./main.tex): manuscript entrypoint
- `sections/`: paper sections
- `figures/`: local copies of manuscript figures
- [references.bib](./references.bib): verified bibliography
- [build.ps1](./build.ps1): local build helper
- `sections/10_biographies.tex`: IEEE-style author biography block without photos

## Build

```powershell
./build.ps1
```

Or directly:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Notes

- The appendix currently holds the midclass and highclass screening tables.
- Figures are copied into this workspace so the manuscript does not depend on ignored analysis output paths.
- Author photos are intentionally not stored in this workspace yet; the manuscript uses `IEEEbiographynophoto` so it remains buildable without final portrait assets.
- This workspace is intentionally self-contained for Overleaf or ScholarOne-style submission packaging.

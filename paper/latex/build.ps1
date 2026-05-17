$ErrorActionPreference = "Stop"

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

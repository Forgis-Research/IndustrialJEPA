#!/bin/bash
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode fig_architecture_sigreg.tex 2>&1 | tail -2
cp fig_domain_overview.pdf fig_tokenization.pdf fig_architecture_ema.pdf fig_architecture_sigreg.pdf ../figures/
echo "All figures copied to ../figures/"

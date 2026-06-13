#!/usr/bin/env bash
# Build the bilingual article PDFs.
#   English  -> pdflatex (Computer Modern), emtl-article-en.pdf
#   Persian  -> xelatex  (xepersian + XB Niloofar), emtl-article-fa.pdf
# Requires: TeX Live with latexmk, xepersian, and the "XB Niloofar" font
# installed (see README). PDFs are written next to the sources.
set -euo pipefail
cd "$(dirname "$0")"

echo "Building English article (pdflatex)..."
latexmk -pdf -interaction=nonstopmode -halt-on-error emtl-article-en.tex

echo "Building Persian article (xelatex)..."
latexmk -xelatex -interaction=nonstopmode -halt-on-error emtl-article-fa.tex

echo "Cleaning auxiliary files..."
latexmk -c emtl-article-en.tex >/dev/null 2>&1 || true
latexmk -c emtl-article-fa.tex >/dev/null 2>&1 || true
rm -f build_en.log build_fa.log

echo "Done: emtl-article-en.pdf, emtl-article-fa.pdf"

#!/bin/bash

# LaTeX compilation script with bibliography
# Run this script to properly compile the paper with references

echo "Starting LaTeX compilation..."

# First pdflatex pass
echo "Running first pdflatex pass..."
pdflatex www2026_paper_enhanced.tex

# Run bibtex to process bibliography
echo "Running bibtex..."
bibtex www2026_paper_enhanced

# Second pdflatex pass (resolve citations)
echo "Running second pdflatex pass..."
pdflatex www2026_paper_enhanced.tex

# Third pdflatex pass (resolve all references)
echo "Running final pdflatex pass..."
pdflatex www2026_paper_enhanced.tex

echo "Compilation complete! Check www2026_paper_enhanced.pdf"

# Clean up auxiliary files (optional)
echo "Cleaning up auxiliary files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls

echo "Done!"

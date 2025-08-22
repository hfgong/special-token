# Makefile for Special Token Magic in Transformers book

# Main LaTeX compiler
LATEX = lualatex
BIBER = biber
LATEX_FLAGS = -shell-escape -interaction=nonstopmode

# Main document
MAIN = main
MAIN_TEX = $(MAIN).tex
MAIN_PDF = $(MAIN).pdf

# Find all figure tex files
FIGURE_SOURCES := $(shell find . -path ./figures -prune -o -name "fig_*.tex" -type f -print)
FIGURE_PDFS := $(FIGURE_SOURCES:.tex=.pdf)

# Default target
all: $(MAIN_PDF)

# Main book compilation with bibliography
$(MAIN_PDF): $(MAIN_TEX) $(FIGURE_PDFS) references.bib
	@echo "Building main document..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)
	$(BIBER) $(MAIN)
	$(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)
	$(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)
	@echo "Build complete!"

# Pattern rule for compiling figure files
fig_%.pdf: fig_%.tex
	@echo "Compiling figure $<..."
	@$(LATEX) $(LATEX_FLAGS) -jobname=$(basename $@) $<
	@rm -f $(basename $@).aux $(basename $@).log $(basename $@).out

# Quick build without bibliography
quick: $(MAIN_TEX) $(FIGURE_PDFS)
	@echo "Quick build..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN_TEX)

# Build only figures
figures: $(FIGURE_PDFS)
	@echo "All figures compiled."

# Clean auxiliary files but keep PDFs
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot
	rm -f $(MAIN).bbl $(MAIN).blg $(MAIN).bcf $(MAIN).run.xml
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz
	rm -f */*.aux */*/*.aux
	find . -name "*.aux" -type f -delete
	find . -name "*.log" -type f -delete
	find . -name "*.out" -type f -delete
	find . -name "fig_*.aux" -type f -delete
	find . -name "fig_*.log" -type f -delete

# Clean everything including PDFs
cleanall: clean
	@echo "Cleaning all generated files..."
	rm -f $(MAIN_PDF)
	rm -f $(FIGURE_PDFS)

# Watch for changes and rebuild
watch:
	@echo "Watching for changes..."
	@while true; do \
		inotifywait -e modify,create,delete -r . --exclude '\.pdf$$|\.aux$$|\.log$$|\.git' 2>/dev/null; \
		make quick; \
	done

# Validate LaTeX syntax
validate:
	@echo "Validating LaTeX syntax..."
	@chktex -q $(MAIN_TEX)

# Count words in the document
wordcount:
	@echo "Counting words..."
	@texcount -inc -total $(MAIN_TEX)

# Generate a draft version with todo notes visible
draft:
	@echo "Building draft version..."
	$(LATEX) $(LATEX_FLAGS) "\def\isdraft{1} \input{$(MAIN_TEX)}"

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the complete book with bibliography (default)"
	@echo "  quick     - Quick build without bibliography"
	@echo "  figures   - Compile all figure files"
	@echo "  clean     - Remove auxiliary files"
	@echo "  cleanall  - Remove all generated files including PDFs"
	@echo "  watch     - Watch for changes and rebuild automatically"
	@echo "  validate  - Check LaTeX syntax"
	@echo "  wordcount - Count words in the document"
	@echo "  draft     - Build draft version with todo notes"
	@echo "  help      - Show this help message"

.PHONY: all quick figures clean cleanall watch validate wordcount draft help
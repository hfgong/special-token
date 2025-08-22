# Book Agent Using Lualatex/Tikz

# Continue/Resume Work

It takes months or even years to write a book, through many agent sessions. Once you get restarted, try to understand the work goal and status by doing the following:

* Check the \*.md files at the top level of the repository to get the big picture.  
* Look at the sub-directories to get a summary of what we have now.  
* List the most recently changed files to know the recent updates and guess the next steps.  
* There might be a handover.md file from the last session, if so read it.

# Directory Structure and Compilation

The latex files are organized in three levels, to keep the context smaller for each LLM call. The structure is part/chapter/section. Each section has a separate file under a chapter directory. We also keep figures in chapter directory.

* Each figure is a standalone latex file without any page numbers etc. It will be compiled into a PDF file and included into the section latex file using \\includegraphics command.  
* We use a **Makefile** to trigger the figure file compilation and the main book latex file compilation.  
* We keep a bib file at the top level directory to keep record of our literature research and to be references in the book text.

# Diagrams and Tables

Use latex **tikz** package for most diagrams, keep each of them in a latex file to be compiled as PDF as mentioned above.

* Always make sure tikz figures compile. A typical broken pattern is multiline text in a node.  
* Make sure nodes/blocks don't overlap with each other, be aware of large nodes/blocks.  
* Use circular and diamond node shapes only for short text, longer text makes them too large with big empty space inside.  
* Be consistent with tikz diagram styles throughout the book.

# Research and Outlining

Always do literature research before writing. Keep the literature in the top level bib file with proper notes.

If starting from scratch, do top level research and top level outlining. For top level research, consider different dimensions like 1\) timeline/historical remarks, 2\) literature review papers, 3\) method categories, 4\) different application domains.

For each part, chapter or section, do research and outline at corresponding level, if no actual work is done yet for that component, for example, no file or just some placeholder file.

Consider diagram/table/formula driven writing. If suited, after writing the outline, try to figure out some ideas about the diagram/table/formula, then write the text body based on the outline and diagrams/tables/formulas.

# Tone and Writing Style

Consider textbook tone and style first if not specified.


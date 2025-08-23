#!/usr/bin/env python3

import os
import re
import subprocess

def find_long_code_blocks():
    """Find all code blocks longer than 100 lines"""
    long_blocks = []
    
    # Get all tex files with code blocks
    result = subprocess.run([
        'find', '/home/hfgong/github/special-token', '-name', '*.tex',
        '-exec', 'grep', '-l', r'\\begin{lstlisting}', '{}', ';'
    ], capture_output=True, text=True)
    
    tex_files = result.stdout.strip().split('\n')
    
    for file_path in tex_files:
        if not file_path:
            continue
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        in_block = False
        block_start = 0
        line_count = 0
        
        for i, line in enumerate(lines):
            if r'\begin{lstlisting}' in line and 'caption=' in line:
                in_block = True
                block_start = i + 1
                line_count = 0
                caption = extract_caption(line)
            elif in_block:
                line_count += 1
                if r'\end{lstlisting}' in line:
                    if line_count > 100:
                        long_blocks.append({
                            'file': file_path,
                            'start': block_start,
                            'end': i + 1,
                            'lines': line_count,
                            'caption': caption
                        })
                    in_block = False
    
    return long_blocks

def extract_caption(line):
    """Extract caption from lstlisting line"""
    match = re.search(r'caption=([^]]+)', line)
    if match:
        return match.group(1).strip('{}').strip()
    return "untitled"

def create_code_file_path(tex_file, caption):
    """Create appropriate code file path"""
    # Extract part and chapter from tex file path
    parts = tex_file.split('/')
    
    # Find part and chapter
    part_dir = None
    chapter_dir = None
    for i, part in enumerate(parts):
        if part.startswith('part'):
            part_dir = part
            if i + 1 < len(parts) and parts[i+1].startswith('chapter'):
                chapter_dir = parts[i+1]
            break
    
    if not part_dir or not chapter_dir:
        return None
    
    # Create filename from caption
    filename = re.sub(r'[^a-zA-Z0-9_\s]', '', caption)
    filename = re.sub(r'\s+', '_', filename).lower()
    if not filename:
        filename = "code_block"
    filename += ".py"
    
    code_dir = f"/home/hfgong/github/special-token/code/{part_dir}/{chapter_dir}"
    os.makedirs(code_dir, exist_ok=True)
    
    return os.path.join(code_dir, filename)

def extract_and_replace_long_blocks():
    """Extract long code blocks and replace with references"""
    long_blocks = find_long_code_blocks()
    
    print(f"Found {len(long_blocks)} long code blocks:")
    for block in long_blocks:
        print(f"  {block['file']}: {block['lines']} lines - {block['caption']}")
    
    for block in long_blocks:
        # Create code file path
        code_file = create_code_file_path(block['file'], block['caption'])
        if not code_file:
            continue
        
        # Extract code content
        with open(block['file'], 'r') as f:
            lines = f.readlines()
        
        # Extract just the Python code (skip lstlisting begin/end)
        code_lines = lines[block['start']:block['end']-1]  # Skip end line
        code_content = ''.join(code_lines)
        
        # Write to code file
        with open(code_file, 'w') as f:
            f.write(code_content)
        
        # Calculate relative path from tex file to code file
        rel_path = os.path.relpath(code_file, os.path.dirname(block['file']))
        
        # Create replacement content
        replacement = f"""The complete implementation is provided in the external code file \\texttt{{{rel_path}}}. Key components include:

\\begin{{lstlisting}}[language=Python, caption=Core structure (see external file for complete implementation)]
# See {rel_path} for the complete implementation
# This shows only the main class structure
{code_lines[0].strip()}
    # ... (complete implementation in external file)
    pass
\\end{{lstlisting}}"""
        
        # Replace in original file
        original_block = ''.join(lines[block['start']-1:block['end']])
        
        with open(block['file'], 'r') as f:
            content = f.read()
        
        new_content = content.replace(original_block, replacement)
        
        with open(block['file'], 'w') as f:
            f.write(new_content)
        
        print(f"Extracted {block['lines']} lines to {code_file}")

if __name__ == "__main__":
    extract_and_replace_long_blocks()
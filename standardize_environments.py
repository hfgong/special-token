#!/usr/bin/env python3

import os
import re
import subprocess

def standardize_code_environments():
    """Standardize example environments to use lstlisting directly"""
    
    # Find all files with example environments
    result = subprocess.run([
        'grep', '-r', r'\\begin{example}', '/home/hfgong/github/special-token',
        '--include=*.tex', '-l'
    ], capture_output=True, text=True)
    
    files_with_examples = result.stdout.strip().split('\n')
    files_modified = 0
    
    for file_path in files_with_examples:
        if not file_path:
            continue
        
        print(f"Processing {file_path}...")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Pattern to match example environments wrapping lstlisting
            pattern = r'\\begin\{example\}\[([^\]]+)\]\s*\\begin\{lstlisting\}\[language=Python\]'
            replacement = r'\\begin{lstlisting}[language=Python, caption=\1]'
            
            new_content = re.sub(pattern, replacement, content)
            
            # Remove corresponding end tags
            pattern2 = r'\\end\{lstlisting\}\s*\\end\{example\}'
            replacement2 = r'\\end{lstlisting}'
            
            new_content = re.sub(pattern2, replacement2, new_content)
            
            if new_content != content:
                with open(file_path, 'w') as f:
                    f.write(new_content)
                files_modified += 1
                print(f"  Modified: {file_path}")
            else:
                print(f"  No changes needed: {file_path}")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print(f"\nStandardization complete. Modified {files_modified} files.")

if __name__ == "__main__":
    standardize_code_environments()
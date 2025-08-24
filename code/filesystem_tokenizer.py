import os
import json
from datetime import datetime

class FileSystemTokenizer:
    def __init__(self, base_tokenizer, sandbox_path="/tmp/sandbox"):
        self.base_tokenizer = base_tokenizer
        self.sandbox_path = sandbox_path
        self.fs_tokens = {
            'read': '[FS_READ]',
            'write': '[FS_WRITE]',
            'append': '[FS_APPEND]',
            'delete': '[FS_DELETE]',
            'mkdir': '[FS_MKDIR]',
            'list': '[FS_LIST]',
            'path': '[PATH]',
            'content': '[CONTENT]',
            'permissions': '[PERMS]',
            'size': '[SIZE]',
            'modified': '[MODIFIED]'
        }
        
    def format_file_operation(self, operation, path, content=None):
        """Format file operation with safety checks"""
        # Ensure path is within sandbox
        full_path = os.path.join(self.sandbox_path, path)
        if not full_path.startswith(self.sandbox_path):
            raise ValueError("Path escapes sandbox")
        
        formatted = [
            self.fs_tokens[operation],
            f"{self.fs_tokens['path']} {path}"
        ]
        
        if operation in ['write', 'append'] and content:
            # Limit content size
            max_size = 1024 * 1024  # 1MB
            if len(content) > max_size:
                content = content[:max_size] + "\n[TRUNCATED]"
            
            formatted.append(f"{self.fs_tokens['content']}")
            formatted.append(content)
        
        return '\n'.join(formatted)
    
    def format_directory_listing(self, path, recursive=False):
        """Format directory listing with metadata"""
        full_path = os.path.join(self.sandbox_path, path)
        
        listing = [
            self.fs_tokens['list'],
            f"{self.fs_tokens['path']} {path}"
        ]
        
        try:
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                stat = os.stat(item_path)
                
                item_info = {
                    'name': item,
                    'type': 'DIR' if os.path.isdir(item_path) else 'FILE',
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                
                listing.append(f"[ITEM] {json.dumps(item_info)}")
                
                if recursive and os.path.isdir(item_path):
                    # Recursively list subdirectories
                    sublisting = self.format_directory_listing(
                        os.path.join(path, item),
                        recursive=True
                    )
                    listing.append(sublisting)
                    
        except PermissionError:
            listing.append("[ERROR] Permission denied")
        except FileNotFoundError:
            listing.append("[ERROR] Path not found")
        
        return '\n'.join(listing)
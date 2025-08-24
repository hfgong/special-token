import json

class DatabaseQueryTokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.db_tokens = {
            'query_start': '[DB_QUERY]',
            'query_end': '[/DB_QUERY]',
            'sql': '[SQL]',
            'nosql': '[NOSQL]',
            'collection': '[COLLECTION]',
            'table': '[TABLE]',
            'result_set': '[RESULT_SET]',
            'row_count': '[ROW_COUNT]',
            'transaction': '[TRANSACTION]',
            'commit': '[COMMIT]',
            'rollback': '[ROLLBACK]'
        }
        
    def format_sql_query(self, query, params=None, transaction=False):
        """Format SQL query with safety tokens"""
        formatted = []
        
        if transaction:
            formatted.append(self.db_tokens['transaction'])
        
        formatted.extend([
            self.db_tokens['query_start'],
            f"{self.db_tokens['sql']} {query}"
        ])
        
        if params:
            # Use parameterized queries for safety
            formatted.append(f"[PARAMS] {params}")
        
        formatted.append(self.db_tokens['query_end'])
        
        return '\n'.join(formatted)
    
    def format_nosql_query(self, collection, operation, filter_doc=None, update_doc=None):
        """Format NoSQL query with tokens"""
        formatted = [
            self.db_tokens['query_start'],
            f"{self.db_tokens['nosql']}",
            f"{self.db_tokens['collection']} {collection}",
            f"[OPERATION] {operation}"
        ]
        
        if filter_doc:
            formatted.append(f"[FILTER] {json.dumps(filter_doc)}")
        
        if update_doc and operation in ['update', 'replace']:
            formatted.append(f"[UPDATE] {json.dumps(update_doc)}")
        
        formatted.append(self.db_tokens['query_end'])
        
        return '\n'.join(formatted)
    
    def parse_query_result(self, result, query_type='sql'):
        """Parse and format query results"""
        formatted = [self.db_tokens['result_set']]
        
        if query_type == 'sql':
            # Format as table
            if result and len(result) > 0:
                # Get column names
                columns = list(result[0].keys())
                formatted.append(f"[COLUMNS] {columns}")
                
                # Add row count
                formatted.append(f"{self.db_tokens['row_count']} {len(result)}")
                
                # Add first few rows
                for i, row in enumerate(result[:5]):
                    formatted.append(f"[ROW_{i}] {row}")
                
                if len(result) > 5:
                    formatted.append(f"... and {len(result) - 5} more rows")
            else:
                formatted.append("[EMPTY_RESULT]")
                
        elif query_type == 'nosql':
            # Format as documents
            formatted.append(f"[DOCUMENT_COUNT] {len(result)}")
            for i, doc in enumerate(result[:3]):
                formatted.append(f"[DOC_{i}] {json.dumps(doc, indent=2)}")
            
            if len(result) > 3:
                formatted.append(f"... and {len(result) - 3} more documents")
        
        return '\n'.join(formatted)
    
    def create_transaction_block(self, queries):
        """Create a transaction block with multiple queries"""
        transaction = [
            self.db_tokens['transaction'],
            "[BEGIN]"
        ]
        
        for i, query in enumerate(queries):
            transaction.append(f"[STEP_{i}]")
            transaction.append(self.format_sql_query(query['sql'], query.get('params')))
            
            # Add validation check
            if 'validate' in query:
                transaction.append(f"[VALIDATE] {query['validate']}")
        
        transaction.extend([
            "[END]",
            self.db_tokens['commit']
        ])
        
        return '\n'.join(transaction)
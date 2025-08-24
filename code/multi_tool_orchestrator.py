import json

class MultiToolOrchestrator:
    def __init__(self, tokenizers):
        self.tokenizers = tokenizers  # Dict of tool-specific tokenizers
        self.orchestration_tokens = {
            'workflow_start': '[WORKFLOW]',
            'workflow_end': '[/WORKFLOW]',
            'parallel': '[PARALLEL]',
            'sequential': '[SEQUENTIAL]',
            'conditional': '[IF]',
            'loop': '[LOOP]',
            'variable': '[VAR]',
            'checkpoint': '[CHECKPOINT]'
        }
        
    def create_workflow(self, workflow_spec):
        """Create a multi-tool workflow from specification"""
        workflow = [self.orchestration_tokens['workflow_start']]
        
        for task in workflow_spec['tasks']:
            if task['type'] == 'parallel':
                workflow.append(self.format_parallel_tasks(task['subtasks']))
            elif task['type'] == 'sequential':
                workflow.append(self.format_sequential_tasks(task['subtasks']))
            elif task['type'] == 'conditional':
                workflow.append(self.format_conditional_task(task))
            elif task['type'] == 'loop':
                workflow.append(self.format_loop_task(task))
            else:
                workflow.append(self.format_single_task(task))
        
        workflow.append(self.orchestration_tokens['workflow_end'])
        
        return '\n'.join(workflow)
    
    def format_parallel_tasks(self, tasks):
        """Format tasks to run in parallel"""
        formatted = [self.orchestration_tokens['parallel']]
        
        for i, task in enumerate(tasks):
            formatted.append(f"[THREAD_{i}]")
            formatted.append(self.format_single_task(task))
        
        formatted.append("[WAIT_ALL]")
        
        return '\n'.join(formatted)
    
    def format_sequential_tasks(self, tasks):
        """Format tasks to run sequentially"""
        formatted = [self.orchestration_tokens['sequential']]
        
        for i, task in enumerate(tasks):
            formatted.append(f"[STEP_{i}]")
            formatted.append(self.format_single_task(task))
            
            # Add checkpoint after each task
            if task.get('checkpoint', False):
                formatted.append(f"{self.orchestration_tokens['checkpoint']} step_{i}")
        
        return '\n'.join(formatted)
    
    def format_conditional_task(self, task):
        """Format conditional task execution"""
        formatted = [
            f"{self.orchestration_tokens['conditional']} {task['condition']}",
            "[THEN]",
            self.format_single_task(task['then_task'])
        ]
        
        if 'else_task' in task:
            formatted.extend([
                "[ELSE]",
                self.format_single_task(task['else_task'])
            ])
        
        formatted.append("[END_IF]")
        
        return '\n'.join(formatted)
    
    def format_loop_task(self, task):
        """Format loop task execution"""
        formatted = [
            f"{self.orchestration_tokens['loop']} {task['iteration_type']}"
        ]
        
        if task['iteration_type'] == 'for':
            formatted.append(f"[RANGE] {task['range']}")
        elif task['iteration_type'] == 'while':
            formatted.append(f"[CONDITION] {task['condition']}")
        elif task['iteration_type'] == 'foreach':
            formatted.append(f"[ITEMS] {task['items']}")
        
        formatted.append("[DO]")
        formatted.append(self.format_single_task(task['body']))
        formatted.append("[END_LOOP]")
        
        return '\n'.join(formatted)
    
    def format_single_task(self, task):
        """Format a single tool task"""
        tool_type = task['tool']
        
        if tool_type in self.tokenizers:
            tokenizer = self.tokenizers[tool_type]
            
            # Use appropriate tokenizer based on tool type
            if tool_type == 'api':
                return tokenizer.format_api_call(**task['params'])
            elif tool_type == 'database':
                return tokenizer.format_sql_query(**task['params'])
            elif tool_type == 'file':
                return tokenizer.format_file_operation(**task['params'])
            elif tool_type == 'web':
                return tokenizer.format_navigation(**task['params'])
            else:
                return f"[UNKNOWN_TOOL] {tool_type}"
        
        return f"[TOOL] {tool_type} {json.dumps(task.get('params', {}))}"
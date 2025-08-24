import inspect
import asyncio

class FunctionCallingTokenizer:
    def __init__(self, base_tokenizer, function_registry):
        self.base_tokenizer = base_tokenizer
        self.function_registry = function_registry
        self.func_tokens = {
            'call': '[FUNC_CALL]',
            'name': '[FUNC_NAME]',
            'args': '[ARGS]',
            'kwargs': '[KWARGS]',
            'return': '[RETURN]',
            'type': '[TYPE]',
            'async': '[ASYNC]',
            'await': '[AWAIT]'
        }
        
    def format_function_call(self, func_name, *args, **kwargs):
        """Format a function call with type information"""
        if func_name not in self.function_registry:
            raise ValueError(f"Function {func_name} not registered")
        
        func_info = self.function_registry[func_name]
        
        formatted = [
            self.func_tokens['call'],
            f"{self.func_tokens['name']} {func_name}"
        ]
        
        # Add type information
        if 'return_type' in func_info:
            formatted.append(f"{self.func_tokens['type']} {func_info['return_type']}")
        
        # Add positional arguments
        if args:
            formatted.append(f"{self.func_tokens['args']} {list(args)}")
        
        # Add keyword arguments
        if kwargs:
            formatted.append(f"{self.func_tokens['kwargs']} {kwargs}")
        
        # Mark if async
        if func_info.get('is_async', False):
            formatted.append(self.func_tokens['async'])
        
        return '\n'.join(formatted)
    
    def validate_function_call(self, func_name, args, kwargs):
        """Validate function call against signature"""
        if func_name not in self.function_registry:
            return False, "Function not found"
        
        func = self.function_registry[func_name]['function']
        sig = inspect.signature(func)
        
        try:
            # Check if arguments match signature
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return True, "Valid call"
        except TypeError as e:
            return False, str(e)
    
    def execute_function_safely(self, func_name, args, kwargs):
        """Execute function with safety checks"""
        # Validate first
        is_valid, message = self.validate_function_call(func_name, args, kwargs)
        
        if not is_valid:
            return {
                'error': True,
                'message': message
            }
        
        func_info = self.function_registry[func_name]
        func = func_info['function']
        
        # Apply safety restrictions
        if func_info.get('requires_confirmation', False):
            # In real implementation, would ask for user confirmation
            pass
        
        if func_info.get('rate_limited', False):
            # Check rate limits
            pass
        
        try:
            # Execute function
            if func_info.get('is_async', False):
                # Handle async execution
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            
            return {
                'error': False,
                'result': result,
                'type': type(result).__name__
            }
            
        except Exception as e:
            return {
                'error': True,
                'message': str(e),
                'type': 'exception'
            }
    
    def create_function_chain(self, chain_spec):
        """Create a chain of function calls with data flow"""
        results = []
        context = {}
        
        for step in chain_spec:
            func_name = step['function']
            
            # Resolve arguments from context
            resolved_args = []
            for arg in step.get('args', []):
                if isinstance(arg, str) and arg.startswith('$'):
                    # Reference to previous result
                    var_name = arg[1:]
                    if var_name in context:
                        resolved_args.append(context[var_name])
                    else:
                        resolved_args.append(arg)
                else:
                    resolved_args.append(arg)
            
            # Execute function
            result = self.execute_function_safely(
                func_name,
                resolved_args,
                step.get('kwargs', {})
            )
            
            # Store result in context
            if 'store_as' in step:
                context[step['store_as']] = result.get('result')
            
            results.append({
                'step': step,
                'result': result
            })
        
        return results
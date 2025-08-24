import json
import re

class ToolInteractionTokenizer:
    def __init__(self, base_tokenizer, available_tools):
        self.base_tokenizer = base_tokenizer
        self.available_tools = available_tools
        self.tool_tokens = {
            'tool_call_start': '[TOOL_CALL]',
            'tool_call_end': '[/TOOL_CALL]',
            'tool_name': '[TOOL]',
            'parameters': '[PARAMS]',
            'result_start': '[RESULT]',
            'result_end': '[/RESULT]',
            'error': '[ERROR]'
        }
        
    def format_tool_call(self, tool_name, parameters):
        """Format a tool call with appropriate tokens"""
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool {tool_name} not available")
        
        formatted = (
            f"{self.tool_tokens['tool_call_start']}\n"
            f"{self.tool_tokens['tool_name']} {tool_name}\n"
            f"{self.tool_tokens['parameters']} {json.dumps(parameters)}\n"
            f"{self.tool_tokens['tool_call_end']}"
        )
        
        return formatted
    
    def parse_model_output_for_tools(self, model_output):
        """Parse model output to extract tool calls"""
        tool_calls = []
        
        # Pattern to match tool calls
        pattern = (
            f"{re.escape(self.tool_tokens['tool_call_start'])}"
            f".*?"
            f"{re.escape(self.tool_tokens['tool_call_end'])}"
        )
        
        matches = re.findall(pattern, model_output, re.DOTALL)
        
        for match in matches:
            # Extract tool name
            tool_pattern = f"{re.escape(self.tool_tokens['tool_name'])}\\s+(\\w+)"
            tool_match = re.search(tool_pattern, match)
            tool_name = tool_match.group(1) if tool_match else None
            
            # Extract parameters
            params_pattern = f"{re.escape(self.tool_tokens['parameters'])}\\s+(\\{{.*?\\}})"
            params_match = re.search(params_pattern, match, re.DOTALL)
            
            if params_match:
                try:
                    parameters = json.loads(params_match.group(1))
                except json.JSONDecodeError:
                    parameters = {}
            else:
                parameters = {}
            
            if tool_name:
                tool_calls.append({
                    'tool': tool_name,
                    'parameters': parameters
                })
        
        return tool_calls
    
    def execute_tool_call(self, tool_name, parameters):
        """Execute a tool call and return formatted result"""
        try:
            # Get the tool function
            tool_func = self.available_tools[tool_name]
            
            # Execute the tool
            result = tool_func(**parameters)
            
            # Format successful result
            formatted_result = (
                f"{self.tool_tokens['result_start']}\n"
                f"{json.dumps(result, indent=2)}\n"
                f"{self.tool_tokens['result_end']}"
            )
            
        except Exception as e:
            # Format error result
            formatted_result = (
                f"{self.tool_tokens['error']}\n"
                f"Tool execution failed: {str(e)}\n"
                f"{self.tool_tokens['result_end']}"
            )
        
        return formatted_result
    
    def create_tool_augmented_prompt(self, user_query, tool_results=None):
        """Create prompt with tool usage context"""
        prompt = f"User Query: {user_query}\n\n"
        
        # Add available tools description
        prompt += "Available tools:\n"
        for tool_name, tool_func in self.available_tools.items():
            if hasattr(tool_func, '__doc__'):
                prompt += f"- {tool_name}: {tool_func.__doc__}\n"
        
        # Add previous tool results if any
        if tool_results:
            prompt += "\nPrevious tool results:\n"
            for result in tool_results:
                prompt += result + "\n"
        
        prompt += "\nResponse (use tools as needed):\n"
        
        return prompt
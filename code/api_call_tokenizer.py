import json

class APICallTokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.api_tokens = {
            'api_start': '[API_CALL]',
            'api_end': '[/API_CALL]',
            'endpoint': '[ENDPOINT]',
            'method': '[METHOD]',
            'headers': '[HEADERS]',
            'body': '[BODY]',
            'response': '[RESPONSE]',
            'status': '[STATUS]',
            'auth': '[AUTH]'
        }
        
    def format_api_call(self, endpoint, method='GET', headers=None, body=None, auth=None):
        """Format an API call with structured tokens"""
        formatted = [self.api_tokens['api_start']]
        
        # Add endpoint
        formatted.append(f"{self.api_tokens['endpoint']} {endpoint}")
        
        # Add HTTP method
        formatted.append(f"{self.api_tokens['method']} {method}")
        
        # Add headers if present
        if headers:
            formatted.append(f"{self.api_tokens['headers']} {json.dumps(headers)}")
        
        # Add authentication if present
        if auth:
            formatted.append(f"{self.api_tokens['auth']} [REDACTED]")
        
        # Add body for POST/PUT requests
        if body and method in ['POST', 'PUT', 'PATCH']:
            formatted.append(f"{self.api_tokens['body']} {json.dumps(body)}")
        
        formatted.append(self.api_tokens['api_end'])
        
        return '\n'.join(formatted)
    
    def parse_api_response(self, response, status_code):
        """Parse and format API response with tokens"""
        formatted = [
            self.api_tokens['response'],
            f"{self.api_tokens['status']} {status_code}"
        ]
        
        if 200 <= status_code < 300:
            # Successful response
            formatted.append(json.dumps(response, indent=2))
        else:
            # Error response
            formatted.append(f"[ERROR] API call failed with status {status_code}")
            if response:
                formatted.append(json.dumps(response, indent=2))
        
        return '\n'.join(formatted)
    
    def create_api_chain(self, api_calls):
        """Create a chain of dependent API calls"""
        chain = []
        results = {}
        
        for i, call in enumerate(api_calls):
            # Substitute variables from previous results
            if 'depends_on' in call:
                for dep_idx, dep_field in call['depends_on'].items():
                    if dep_idx in results:
                        # Replace placeholder with actual value
                        placeholder = f"${{result_{dep_idx}.{dep_field}}}"
                        actual_value = results[dep_idx].get(dep_field, '')
                        
                        # Update endpoint or body with actual value
                        if 'endpoint' in call:
                            call['endpoint'] = call['endpoint'].replace(
                                placeholder, str(actual_value)
                            )
                        if 'body' in call:
                            call['body'] = json.loads(
                                json.dumps(call['body']).replace(
                                    placeholder, str(actual_value)
                                )
                            )
            
            # Format the API call
            formatted_call = self.format_api_call(**call)
            chain.append(formatted_call)
            
            # Simulate execution and store result
            # In real implementation, this would actually call the API
            results[i] = {'simulated': 'result'}
        
        return chain
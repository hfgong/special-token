class WebScrapingTokenizer:
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.web_tokens = {
            'navigate': '[WEB_NAVIGATE]',
            'click': '[WEB_CLICK]',
            'type': '[WEB_TYPE]',
            'select': '[WEB_SELECT]',
            'extract': '[WEB_EXTRACT]',
            'screenshot': '[WEB_SCREENSHOT]',
            'wait': '[WEB_WAIT]',
            'selector': '[SELECTOR]',
            'xpath': '[XPATH]',
            'url': '[URL]',
            'element': '[ELEMENT]'
        }
        
    def format_navigation(self, url, wait_for=None):
        """Format web navigation action"""
        formatted = [
            self.web_tokens['navigate'],
            f"{self.web_tokens['url']} {url}"
        ]
        
        if wait_for:
            formatted.append(f"{self.web_tokens['wait']} {wait_for}")
        
        return '\n'.join(formatted)
    
    def format_interaction(self, action, selector, value=None):
        """Format web interaction action"""
        formatted = [
            self.web_tokens[action],
            f"{self.web_tokens['selector']} {selector}"
        ]
        
        if value:
            formatted.append(f"[VALUE] {value}")
        
        return '\n'.join(formatted)
    
    def format_extraction(self, selectors, extract_type='text'):
        """Format data extraction from web page"""
        formatted = [
            self.web_tokens['extract'],
            f"[EXTRACT_TYPE] {extract_type}"
        ]
        
        for name, selector in selectors.items():
            formatted.append(f"[FIELD] {name} -> {selector}")
        
        return '\n'.join(formatted)
    
    def create_scraping_workflow(self, steps):
        """Create a complete web scraping workflow"""
        workflow = []
        
        for i, step in enumerate(steps):
            workflow.append(f"[STEP_{i}]")
            
            if step['action'] == 'navigate':
                workflow.append(self.format_navigation(
                    step['url'],
                    step.get('wait_for')
                ))
            elif step['action'] in ['click', 'type', 'select']:
                workflow.append(self.format_interaction(
                    step['action'],
                    step['selector'],
                    step.get('value')
                ))
            elif step['action'] == 'extract':
                workflow.append(self.format_extraction(
                    step['selectors'],
                    step.get('type', 'text')
                ))
            elif step['action'] == 'screenshot':
                workflow.append(f"{self.web_tokens['screenshot']} {step.get('filename', 'screenshot.png')}")
            
            # Add validation if specified
            if 'validate' in step:
                workflow.append(f"[VALIDATE] {step['validate']}")
        
        return '\n'.join(workflow)
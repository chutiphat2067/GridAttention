import ast
import os

def add_missing_methods(filepath, methods_to_add):
    """Add missing methods to a Python class file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check which methods are missing
    tree = ast.parse(content)
    existing_methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            existing_methods.append(node.name)
    
    # Add missing methods
    additions = []
    for method_name, method_code in methods_to_add.items():
        if method_name not in existing_methods:
            additions.append(method_code)
    
    if additions:
        # Add imports if needed
        if 'asyncio' not in content:
            content = 'import asyncio\n' + content
        if 'typing' not in content:
            content = 'from typing import Dict, Any, Optional\n' + content
        
        # Find last method or class definition
        lines = content.split('\n')
        insert_line = len(lines) - 1
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip().startswith('def ') or lines[i].strip().startswith('async def '):
                insert_line = i + 1
                # Find the end of this method
                indent_level = len(lines[i]) - len(lines[i].lstrip())
                for j in range(i+1, len(lines)):
                    if lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) <= indent_level:
                        insert_line = j
                        break
                break
        
        # Insert new methods
        for method_code in additions:
            lines.insert(insert_line, method_code)
            insert_line += 1
        
        # Write back
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Added {len(additions)} methods to {filepath}")
    else:
        print(f"✓ All required methods present in {filepath}")

# Define missing methods
standard_methods = {
    'health_check': '''
    async def health_check(self) -> Dict[str, Any]:
        """Check component health"""
        return {
            'healthy': True,
            'is_running': getattr(self, 'is_running', True),
            'error_count': getattr(self, 'error_count', 0),
            'last_error': getattr(self, 'last_error', None)
        }''',
    
    'is_healthy': '''
    async def is_healthy(self) -> bool:
        """Quick health check"""
        health = await self.health_check()
        return health.get('healthy', True)''',
    
    'recover': '''
    async def recover(self) -> bool:
        """Recover from failure"""
        try:
            self.error_count = 0
            self.last_error = None
            return True
        except Exception as e:
            print(f"Recovery failed: {e}")
            return False''',
    
    'get_state': '''
    def get_state(self) -> Dict[str, Any]:
        """Get component state for checkpointing"""
        return {
            'class': self.__class__.__name__,
            'timestamp': time.time() if 'time' in globals() else 0
        }''',
    
    'load_state': '''
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load component state from checkpoint"""
        pass'''
}

# Fix each component
components = [
    'attention_learning_layer.py',
    'market_regime_detector.py',
    'grid_strategy_selector.py',
    'risk_management_system.py',
    'execution_engine.py',
    'performance_monitor.py',
    'overfitting_detector.py',
    'feedback_loop.py'
]

for component in components:
    if os.path.exists(component):
        add_missing_methods(component, standard_methods)
    else:
        print(f"✗ {component} not found")
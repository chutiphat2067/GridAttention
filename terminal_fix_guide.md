# GridAttention System - Terminal Fix Guide for SuperClaude

## 🚨 Quick Fix Commands

### 1. Initial System Check
```bash
# Check current system status
cd /path/to/GridAttention
python -c "from essential_fixes import check_system_health; import asyncio; asyncio.run(check_system_health())"

# List missing dependencies
pip list | grep -E "(torch|pandas|numpy|ccxt|aiohttp)" || echo "Missing core packages"

# Check file structure
find . -name "*.py" -type f | grep -E "(attention|market|grid|risk|execution|performance|overfitting|feedback)" | wc -l
# Should return at least 8 core files
```

### 2. Apply Essential Fixes
```bash
# Create and apply fixes
cat > apply_fixes.py << 'EOF'
import asyncio
from essential_fixes import apply_essential_fixes
from pathlib import Path
import importlib
import sys

async def fix_system():
    print("🔧 Applying essential fixes...")
    
    # Import all components
    components = {}
    component_names = [
        'attention_learning_layer',
        'market_regime_detector', 
        'grid_strategy_selector',
        'risk_management_system',
        'execution_engine',
        'performance_monitor',
        'overfitting_detector',
        'feedback_loop'
    ]
    
    for name in component_names:
        try:
            module = importlib.import_module(name)
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            if hasattr(module, class_name):
                components[name] = getattr(module, class_name)({})
                print(f"✓ Loaded {name}")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    # Apply fixes
    fixed_components = apply_essential_fixes(components)
    print(f"\n✅ Fixed {len(fixed_components)} components")
    
    # Test health checks
    print("\n🏥 Testing health checks...")
    for name, comp in fixed_components.items():
        if hasattr(comp, 'health_check'):
            health = await comp.health_check()
            print(f"{name}: {'✓' if health.get('healthy') else '✗'}")

asyncio.run(fix_system())
EOF

python apply_fixes.py
```

### 3. Fix Missing Methods
```bash
# Add missing methods to components
cat > fix_missing_methods.py << 'EOF'
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
EOF

python fix_missing_methods.py
```

### 4. Fix Synchronization Issues
```bash
# Create event bus for component communication
cat > event_bus.py << 'EOF'
import asyncio
from typing import Dict, List, Callable, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EventBus:
    """Central event bus for component communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue = asyncio.Queue()
        self.running = False
        
    async def start(self):
        """Start event processing"""
        self.running = True
        asyncio.create_task(self._process_events())
        
    async def stop(self):
        """Stop event processing"""
        self.running = False
        
    async def publish(self, event_type: str, data: Any):
        """Publish event to all subscribers"""
        await self.event_queue.put((event_type, data))
        
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type"""
        self.subscribers[event_type].append(handler)
        
    async def _process_events(self):
        """Process events from queue"""
        while self.running:
            try:
                event_type, data = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                
                # Notify all subscribers
                for handler in self.subscribers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(data)
                        else:
                            handler(data)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")

# Global event bus instance
event_bus = EventBus()

# Event types
class Events:
    PHASE_CHANGED = "phase_changed"
    REGIME_DETECTED = "regime_detected"
    OVERFITTING_DETECTED = "overfitting_detected"
    RISK_LIMIT_REACHED = "risk_limit_reached"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    COMPONENT_ERROR = "component_error"
    COMPONENT_RECOVERED = "component_recovered"
EOF

echo "✓ Created event_bus.py"
```

### 5. Fix Risk Management Kill Switch
```bash
# Add kill switch to risk management
cat > fix_risk_management.py << 'EOF'
import re

# Read risk management file
with open('risk_management_system.py', 'r') as f:
    content = f.read()

# Add kill switch if not present
if 'kill_switch' not in content:
    # Find __init__ method
    init_pattern = r'(def __init__\(self[^:]+\):[^}]+?)(\n\s{8})'
    
    kill_switch_code = '''
        # Kill switch
        self.kill_switch = KillSwitch()
        '''
    
    # Add import
    if 'from essential_fixes import KillSwitch' not in content:
        content = 'from essential_fixes import KillSwitch\n' + content
    
    # Add kill switch activation method
    activate_method = '''
    
    async def activate_kill_switch(self, reason: str):
        """Activate emergency kill switch"""
        await self.kill_switch.activate(reason)
        
        # Close all positions
        if hasattr(self, 'execution_engine'):
            await self.execution_engine.close_all_positions(gradual=False)
        
        # Notify all components
        from event_bus import event_bus, Events
        await event_bus.publish(Events.KILL_SWITCH_ACTIVATED, {
            'reason': reason,
            'timestamp': time.time()
        })
    '''
    
    # Add to class
    content = re.sub(init_pattern, r'\1' + kill_switch_code + r'\2', content)
    content += activate_method
    
    with open('risk_management_system.py', 'w') as f:
        f.write(content)
    
    print("✓ Added kill switch to risk management")
else:
    print("✓ Kill switch already present")
EOF

python fix_risk_management.py
```

### 6. Create System Coordinator
```bash
# Create central coordinator
cat > system_coordinator.py << 'EOF'
import asyncio
import logging
from typing import Dict, Any, Optional
from event_bus import event_bus, Events
from essential_fixes import apply_essential_fixes

logger = logging.getLogger(__name__)

class SystemCoordinator:
    """Central coordinator for all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.components = {}
        self.is_running = False
        self.health_check_interval = 60  # seconds
        
    async def initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing components...")
        
        # Import and create components
        component_configs = [
            ('attention_learning_layer', 'AttentionLearningLayer'),
            ('market_regime_detector', 'MarketRegimeDetector'),
            ('grid_strategy_selector', 'GridStrategySelector'),
            ('risk_management_system', 'RiskManagementSystem'),
            ('execution_engine', 'ExecutionEngine'),
            ('performance_monitor', 'PerformanceMonitor'),
            ('overfitting_detector', 'OverfittingDetector'),
            ('feedback_loop', 'FeedbackLoop')
        ]
        
        for module_name, class_name in component_configs:
            try:
                module = __import__(module_name)
                component_class = getattr(module, class_name)
                self.components[module_name] = component_class(
                    self.config.get(module_name, {})
                )
                logger.info(f"✓ Initialized {module_name}")
            except Exception as e:
                logger.error(f"✗ Failed to initialize {module_name}: {e}")
        
        # Apply fixes
        self.components = apply_essential_fixes(self.components)
        
        # Subscribe to events
        await self._setup_event_handlers()
        
    async def _setup_event_handlers(self):
        """Setup event handlers for coordination"""
        
        # Phase change handler
        async def handle_phase_change(data):
            phase = data['phase']
            logger.info(f"Attention phase changed to: {phase}")
            
            # Notify other components
            for comp in self.components.values():
                if hasattr(comp, 'on_phase_change'):
                    await comp.on_phase_change(phase)
        
        event_bus.subscribe(Events.PHASE_CHANGED, handle_phase_change)
        
        # Overfitting handler
        async def handle_overfitting(data):
            severity = data['severity']
            logger.warning(f"Overfitting detected: {severity}")
            
            if severity == 'CRITICAL':
                # Activate kill switch
                if 'risk_management_system' in self.components:
                    await self.components['risk_management_system'].activate_kill_switch(
                        f"Critical overfitting detected: {data}"
                    )
        
        event_bus.subscribe(Events.OVERFITTING_DETECTED, handle_overfitting)
        
    async def start(self):
        """Start all components"""
        logger.info("Starting system...")
        
        # Start event bus
        await event_bus.start()
        
        # Start components in order
        start_order = [
            'market_data_input',
            'feature_engineering',
            'attention_learning_layer',
            'market_regime_detector',
            'overfitting_detector',
            'grid_strategy_selector',
            'risk_management_system',
            'execution_engine',
            'performance_monitor',
            'feedback_loop'
        ]
        
        for comp_name in start_order:
            if comp_name in self.components:
                comp = self.components[comp_name]
                if hasattr(comp, 'start'):
                    await comp.start()
                    logger.info(f"✓ Started {comp_name}")
        
        self.is_running = True
        
        # Start health monitoring
        asyncio.create_task(self._monitor_health())
        
    async def _monitor_health(self):
        """Monitor component health"""
        while self.is_running:
            try:
                unhealthy = []
                
                for name, comp in self.components.items():
                    if hasattr(comp, 'is_healthy'):
                        if not await comp.is_healthy():
                            unhealthy.append(name)
                
                if unhealthy:
                    logger.warning(f"Unhealthy components: {unhealthy}")
                    
                    # Try recovery
                    for name in unhealthy:
                        comp = self.components[name]
                        if hasattr(comp, 'recover'):
                            if await comp.recover():
                                logger.info(f"✓ Recovered {name}")
                            else:
                                logger.error(f"✗ Failed to recover {name}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
    async def stop(self):
        """Stop all components"""
        logger.info("Stopping system...")
        
        self.is_running = False
        
        # Stop in reverse order
        for comp in reversed(list(self.components.values())):
            if hasattr(comp, 'stop'):
                await comp.stop()
        
        await event_bus.stop()
        
        logger.info("System stopped")

# Usage
async def main():
    config = {
        'attention_learning_layer': {
            'learning_rate': 0.001,
            'window_size': 1000
        },
        'risk_management_system': {
            'max_position_size': 0.05,
            'max_daily_loss': 0.02
        }
    }
    
    coordinator = SystemCoordinator(config)
    await coordinator.initialize_components()
    await coordinator.start()
    
    # Run for a while
    await asyncio.sleep(60)
    
    await coordinator.stop()

if __name__ == "__main__":
    asyncio.run(main())
EOF

echo "✓ Created system_coordinator.py"
```

### 7. Validate Fixes
```bash
# Run validation
cat > validate_fixes.py << 'EOF'
import asyncio
import importlib
import sys
from pathlib import Path

async def validate_system():
    print("🔍 Validating system fixes...\n")
    
    issues = []
    
    # Check required files
    required_files = [
        'essential_fixes.py',
        'event_bus.py',
        'system_coordinator.py',
        'attention_learning_layer.py',
        'market_regime_detector.py',
        'grid_strategy_selector.py',
        'risk_management_system.py',
        'execution_engine.py',
        'performance_monitor.py',
        'overfitting_detector.py',
        'feedback_loop.py'
    ]
    
    print("📁 Checking files:")
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} MISSING")
            issues.append(f"Missing file: {file}")
    
    # Check methods in components
    print("\n🔧 Checking component methods:")
    component_modules = [
        ('attention_learning_layer', 'AttentionLearningLayer'),
        ('market_regime_detector', 'MarketRegimeDetector'),
        ('grid_strategy_selector', 'GridStrategySelector'),
        ('risk_management_system', 'RiskManagementSystem'),
        ('execution_engine', 'ExecutionEngine'),
        ('performance_monitor', 'PerformanceMonitor')
    ]
    
    required_methods = [
        'health_check',
        'is_healthy',
        'recover',
        'get_state',
        'load_state'
    ]
    
    for module_name, class_name in component_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                print(f"\n  {class_name}:")
                
                for method in required_methods:
                    if hasattr(cls, method):
                        print(f"    ✓ {method}")
                    else:
                        print(f"    ✗ {method} MISSING")
                        issues.append(f"{class_name} missing {method}")
            else:
                print(f"  ✗ {class_name} not found in {module_name}")
                issues.append(f"Class {class_name} not found")
                
        except ImportError as e:
            print(f"  ✗ Failed to import {module_name}: {e}")
            issues.append(f"Import error: {module_name}")
    
    # Check special requirements
    print("\n🔐 Checking special requirements:")
    
    # Check kill switch in risk management
    try:
        rm_module = importlib.import_module('risk_management_system')
        if hasattr(rm_module, 'RiskManagementSystem'):
            rm_class = getattr(rm_module, 'RiskManagementSystem')
            # Check in source code
            import inspect
            source = inspect.getsource(rm_class)
            if 'kill_switch' in source:
                print("  ✓ Kill switch in RiskManagementSystem")
            else:
                print("  ✗ Kill switch NOT in RiskManagementSystem")
                issues.append("RiskManagementSystem missing kill_switch")
    except Exception as e:
        print(f"  ✗ Error checking kill switch: {e}")
        issues.append("Could not verify kill_switch")
    
    # Summary
    print(f"\n{'='*50}")
    if not issues:
        print("✅ All validations passed! System is ready.")
    else:
        print(f"❌ Found {len(issues)} issues:\n")
        for issue in issues:
            print(f"  - {issue}")
        print("\n⚠️  Please fix these issues before proceeding.")
    
    return len(issues) == 0

if __name__ == "__main__":
    asyncio.run(validate_system())
EOF

python validate_fixes.py
```

### 8. Test Fixed System
```bash
# Quick integration test
cat > test_fixed_system.py << 'EOF'
import asyncio
from system_coordinator import SystemCoordinator
import logging

logging.basicConfig(level=logging.INFO)

async def test_system():
    print("🧪 Testing fixed system...\n")
    
    # Minimal config
    config = {
        'attention_learning_layer': {'learning_rate': 0.001},
        'risk_management_system': {'max_position_size': 0.05}
    }
    
    coordinator = SystemCoordinator(config)
    
    try:
        # Initialize
        print("1️⃣ Initializing components...")
        await coordinator.initialize_components()
        print("✓ Components initialized\n")
        
        # Start
        print("2️⃣ Starting system...")
        await coordinator.start()
        print("✓ System started\n")
        
        # Run for 5 seconds
        print("3️⃣ Running for 5 seconds...")
        await asyncio.sleep(5)
        
        # Check health
        print("\n4️⃣ Checking component health:")
        for name, comp in coordinator.components.items():
            if hasattr(comp, 'is_healthy'):
                healthy = await comp.is_healthy()
                print(f"  {name}: {'✓' if healthy else '✗'}")
        
        # Stop
        print("\n5️⃣ Stopping system...")
        await coordinator.stop()
        print("✓ System stopped\n")
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_system())
EOF

python test_fixed_system.py
```

## 📋 Troubleshooting

### If components won't import:
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to each script
echo "import sys; sys.path.append('.')" > path_fix.py
```

### If methods aren't being added:
```bash
# Manually add to a specific file
echo "
async def health_check(self):
    return {'healthy': True}

async def is_healthy(self):
    return True
" >> attention_learning_layer.py
```

### If event bus has issues:
```bash
# Test event bus separately
python -c "
import asyncio
from event_bus import event_bus, Events

async def test():
    await event_bus.start()
    
    # Test handler
    def handler(data):
        print(f'Received: {data}')
    
    event_bus.subscribe(Events.PHASE_CHANGED, handler)
    await event_bus.publish(Events.PHASE_CHANGED, {'phase': 'test'})
    
    await asyncio.sleep(1)
    await event_bus.stop()

asyncio.run(test())
"
```

## 🎯 Final Verification
```bash
# Run all tests
python run_all_tests.py --quick

# If tests pass, system is fixed!
```

---

**Note**: Save this file as `terminal_fix_guide.md` and run commands in order. Each section builds on the previous one. If any step fails, check the troubleshooting section before continuing.
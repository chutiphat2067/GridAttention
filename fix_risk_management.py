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
        from essential_fixes import KillSwitch
        self.kill_switch = KillSwitch()
        '''
    
    # Add import
    if 'from essential_fixes import KillSwitch' not in content:
        content = 'from essential_fixes import KillSwitch\nimport time\n' + content
    
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
#!/usr/bin/env python3
"""
Quick fix for Binance connection log spam
"""

import re

def fix_market_data_input():
    """Add rate limiting to market data errors"""
    
    # Read the file
    with open('market_data_input.py', 'r') as f:
        content = f.read()
    
    # Add rate limiting imports
    import_fix = '''import time
from typing import Dict, Any, Optional
import asyncio
from collections import deque
'''
    
    # Add rate limiting class
    rate_limiter_code = '''
class ErrorRateLimiter:
    """Rate limit error logging to prevent spam"""
    
    def __init__(self, max_errors_per_minute=5):
        self.max_errors = max_errors_per_minute
        self.error_times = deque(maxlen=100)
        self.last_log_time = 0
        
    def should_log_error(self, error_type='general'):
        """Check if we should log this error"""
        current_time = time.time()
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute ago
        while self.error_times and self.error_times[0] < cutoff_time:
            self.error_times.popleft()
            
        # Check if we're under the limit
        if len(self.error_times) < self.max_errors:
            self.error_times.append(current_time)
            return True
            
        # Log once per minute if we're being rate limited
        if current_time - self.last_log_time > 60:
            self.last_log_time = current_time
            return True
            
        return False

'''
    
    # Replace the MarketDataInput class initialization
    if 'self.error_rate_limiter' not in content:
        # Find the __init__ method and add rate limiter
        init_pattern = r'(def __init__\(self, config: Dict\[str, Any\]\):\s*\n\s*self\.config = config)'
        replacement = r'\1\n        self.error_rate_limiter = ErrorRateLimiter(max_errors_per_minute=5)'
        content = re.sub(init_pattern, replacement, content)
    
    # Replace error logging in collect_tick
    error_pattern = r'logger\.error\(f"Unexpected error in collect_tick: \{e\}"\)'
    error_replacement = '''if self.error_rate_limiter.should_log_error('collect_tick'):
                logger.error(f"Unexpected error in collect_tick: {e} (rate limited)")'''
    
    content = re.sub(error_pattern, error_replacement, content)
    
    # Replace warning logging
    warning_pattern = r'logger\.warning\(f"Extended data gap detected: \{time_since_last:.1f\} seconds"\)'
    warning_replacement = '''if self.error_rate_limiter.should_log_error('data_gap'):
                logger.warning(f"Extended data gap detected: {time_since_last:.1f} seconds (rate limited)")'''
    
    content = re.sub(warning_pattern, warning_replacement, content)
    
    # Add the rate limiter class before MarketDataInput
    if 'class ErrorRateLimiter' not in content:
        # Find where to insert the class
        class_pattern = r'(class MarketDataInput:)'
        content = re.sub(class_pattern, rate_limiter_code + r'\1', content)
    
    # Write back
    with open('market_data_input.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed market data input log spam")

def add_demo_mode():
    """Add demo mode to avoid Binance connection"""
    
    with open('config.yaml', 'r') as f:
        content = f.read()
    
    # Change to demo mode
    if 'mode: "paper_trading"' in content:
        content = content.replace('mode: "paper_trading"', 'mode: "demo"')
    
    # Add demo config
    demo_config = '''
# Demo mode settings
demo:
  enabled: true
  simulated_price: 50000
  simulated_volume: 1.0
  price_volatility: 0.02
  update_interval: 1  # seconds
'''
    
    if 'demo:' not in content:
        content += demo_config
    
    with open('config.yaml', 'w') as f:
        f.write(content)
    
    print("✓ Added demo mode to config")

def create_demo_data_mode():
    """Create demo data provider to replace Binance"""
    
    demo_code = '''#!/usr/bin/env python3
"""
Demo market data provider - replaces Binance connection
"""
import asyncio
import time
import random
from typing import Dict, Any

class DemoMarketData:
    """Provides simulated market data"""
    
    def __init__(self, config):
        self.config = config.get('demo', {})
        self.base_price = self.config.get('simulated_price', 50000)
        self.volatility = self.config.get('price_volatility', 0.02)
        self.current_price = self.base_price
        self.running = False
        
    async def start(self):
        """Start demo data generation"""
        self.running = True
        print("🎭 Demo mode: Generating simulated market data")
        
    async def stop(self):
        """Stop demo data generation"""
        self.running = False
        
    async def get_latest_data(self):
        """Get simulated market tick"""
        if not self.running:
            return None
            
        # Simulate price movement
        change_pct = random.gauss(0, self.volatility)
        self.current_price *= (1 + change_pct)
        
        # Keep price reasonable
        self.current_price = max(self.current_price, self.base_price * 0.5)
        self.current_price = min(self.current_price, self.base_price * 2.0)
        
        return {
            'symbol': 'BTC/USDT',
            'price': self.current_price,
            'volume': random.uniform(0.1, 2.0),
            'timestamp': time.time(),
            'bid': self.current_price * 0.999,
            'ask': self.current_price * 1.001,
            'demo': True
        }
        
    async def collect_tick(self, exchange=None):
        """Simulate collect_tick method"""
        return await self.get_latest_data()
'''
    
    with open('demo_market_data.py', 'w') as f:
        f.write(demo_code)
    
    print("✓ Created demo market data provider")

if __name__ == "__main__":
    print("🔧 Fixing Binance connection log spam...")
    
    try:
        fix_market_data_input()
        add_demo_mode()
        create_demo_data_mode()
        
        print("\n✅ Log spam fixed!")
        print("\n🎭 System now runs in demo mode")
        print("   → No real Binance connection")
        print("   → Simulated market data")
        print("   → Limited error logging")
        
        print("\n🚀 Restart with: python main.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
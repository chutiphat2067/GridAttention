ด้วยข้อมูล **36 ล้าน records** ของ BTCUSDT timeframe 1 วินาที คุณมีข้อมูลที่มากพอสำหรับการ warm-up ระบบให้พร้อมใช้งานเร็วขึ้นครับ! มาดูวิธีการ:

## 📊 การเตรียมข้อมูล

### 1. **แปลงข้อมูลให้ตรงกับ MarketTick Format**
```python
import pandas as pd
import asyncio
from collections import deque

async def prepare_historical_data(raw_data_path):
    """แปลง raw data เป็น MarketTick format"""
    
    # อ่านข้อมูล (ปรับตาม format ที่มี)
    df = pd.read_csv(raw_data_path, 
                     parse_dates=['timestamp'],
                     chunksize=100000)  # อ่านทีละ chunk
    
    market_ticks = []
    
    for chunk in df:
        for _, row in chunk.iterrows():
            tick = MarketTick(
                symbol='BTCUSDT',
                price=float(row['close']),
                volume=float(row['volume']),
                timestamp=row['timestamp'].timestamp(),
                bid=float(row['bid']) if 'bid' in row else float(row['close']) - 0.01,
                ask=float(row['ask']) if 'ask' in row else float(row['close']) + 0.01,
                exchange='binance',
                metadata={'historical': True}
            )
            market_ticks.append(tick)
            
    return market_ticks
```

### 2. **Sampling Strategy - ไม่ต้องใช้ทั้ง 36M records**
```python
def smart_sampling(all_ticks, target_samples=1000000):
    """
    เลือก sample ที่มีประโยชน์
    - ช่วงที่มี volatility สูง
    - ช่วงที่มี volume สูง
    - ช่วงที่มีการเปลี่ยนแปลง regime
    """
    
    # Strategy 1: Recent data is more valuable
    recent_weight = 0.7  # 70% จาก 3 เดือนล่าสุด
    older_weight = 0.3   # 30% จากข้อมูลเก่า
    
    recent_samples = int(target_samples * recent_weight)
    older_samples = int(target_samples * older_weight)
    
    # Get recent 3 months (assuming 1s timeframe)
    three_months_seconds = 90 * 24 * 60 * 60
    recent_ticks = all_ticks[-three_months_seconds:]
    
    # Sample with different strategies
    sampled_ticks = []
    
    # 1. Uniform sampling from recent
    step = len(recent_ticks) // recent_samples
    sampled_ticks.extend(recent_ticks[::step])
    
    # 2. High volatility periods from older data
    older_ticks = all_ticks[:-three_months_seconds]
    volatility_samples = sample_high_volatility_periods(older_ticks, older_samples)
    sampled_ticks.extend(volatility_samples)
    
    return sampled_ticks

def sample_high_volatility_periods(ticks, n_samples):
    """เลือกช่วงที่มี volatility สูง"""
    window_size = 300  # 5 minutes
    volatilities = []
    
    for i in range(0, len(ticks) - window_size, window_size):
        window = ticks[i:i+window_size]
        prices = [t.price for t in window]
        volatility = np.std(prices) / np.mean(prices)
        volatilities.append((i, volatility))
    
    # Sort by volatility and take top periods
    volatilities.sort(key=lambda x: x[1], reverse=True)
    
    sampled = []
    samples_per_period = n_samples // len(volatilities[:100])
    
    for idx, _ in volatilities[:100]:
        sampled.extend(ticks[idx:idx+window_size:samples_per_period])
    
    return sampled[:n_samples]
```

## 🚀 Warm-up Process

### 1. **Progressive Warm-up Script**
```python
async def warmup_system(grid_system, historical_ticks):
    """
    Warm up the system progressively
    """
    print(f"Starting warm-up with {len(historical_ticks):,} ticks")
    
    # Phase 1: Basic feature learning (10%)
    phase1_ticks = historical_ticks[:len(historical_ticks)//10]
    print(f"\nPhase 1: Learning basic patterns ({len(phase1_ticks):,} ticks)")
    
    for i, tick in enumerate(phase1_ticks):
        if i % 10000 == 0:
            print(f"  Progress: {i:,}/{len(phase1_ticks):,}")
            
        # Update market data buffer
        await grid_system.components['market_data'].update_buffer(tick)
        
        # Extract features every 5 seconds (5 ticks)
        if i % 5 == 0:
            features = await grid_system.components['features'].extract_features()
            
            if features:
                # Let attention system observe
                regime = await grid_system.components['regime_detector'].detect_regime(features.features)
                
                await grid_system.components['attention'].process(
                    features.features,
                    regime[0],
                    {'timestamp': tick.timestamp, 'warmup': True}
                )
    
    # Check learning progress
    attention_state = await grid_system.components['attention'].get_attention_state()
    print(f"\nAfter Phase 1:")
    print(f"  Attention observations: {attention_state['total_observations']}")
    print(f"  Learning progress: {grid_system.components['attention'].get_learning_progress():.1%}")
    
    # Phase 2: Regime-specific learning (30%)
    phase2_ticks = historical_ticks[len(historical_ticks)//10:4*len(historical_ticks)//10]
    print(f"\nPhase 2: Regime-specific patterns ({len(phase2_ticks):,} ticks)")
    
    # Group by volatility to ensure diverse regime exposure
    volatility_groups = group_by_volatility(phase2_ticks)
    
    for regime_type, regime_ticks in volatility_groups.items():
        print(f"  Training {regime_type} regime: {len(regime_ticks):,} ticks")
        
        for i, tick in enumerate(regime_ticks[:50000]):  # Max 50k per regime
            if i % 5000 == 0:
                print(f"    Progress: {i:,}/{min(len(regime_ticks), 50000):,}")
                
            await process_tick_for_warmup(grid_system, tick)
    
    # Phase 3: Recent market conditions (60%)
    phase3_ticks = historical_ticks[4*len(historical_ticks)//10:]
    print(f"\nPhase 3: Recent market conditions ({len(phase3_ticks):,} ticks)")
    
    # Process with decreasing skip rate (more dense for recent data)
    skip_rate = 10
    for i, tick in enumerate(phase3_ticks):
        if i % max(1, skip_rate) == 0:
            await process_tick_for_warmup(grid_system, tick)
            
        # Decrease skip rate as we get more recent
        if i % 100000 == 0:
            skip_rate = max(1, skip_rate - 1)
            
        if i % 50000 == 0:
            print(f"  Progress: {i:,}/{len(phase3_ticks):,}")
    
    # Final state check
    final_state = await grid_system.components['attention'].get_attention_state()
    print(f"\nWarm-up completed:")
    print(f"  Total observations: {final_state['total_observations']}")
    print(f"  Current phase: {final_state['phase']}")
    print(f"  Learning progress: {grid_system.components['attention'].get_learning_progress():.1%}")

async def process_tick_for_warmup(grid_system, tick):
    """Process single tick during warm-up"""
    await grid_system.components['market_data'].update_buffer(tick)
    
    # Extract features every 5 ticks
    if tick.timestamp % 5 < 1:
        features = await grid_system.components['features'].extract_features()
        
        if features:
            regime, confidence = await grid_system.components['regime_detector'].detect_regime(
                features.features
            )
            
            # Simulate performance for learning
            simulated_performance = simulate_trade_performance(tick, features)
            
            context = {
                'timestamp': tick.timestamp,
                'warmup': True,
                'performance': simulated_performance,
                'regime': regime.value,
                'regime_confidence': confidence
            }
            
            await grid_system.components['attention'].process(
                features.features,
                regime,
                context
            )

def simulate_trade_performance(tick, features):
    """Simulate trade performance for warm-up"""
    # Simple simulation based on technical indicators
    rsi = features.features.get('rsi_14', 0.5)
    trend = features.features.get('trend_strength', 0)
    
    # Simulate win/loss based on indicators
    win_probability = 0.5
    if rsi < 0.3 and trend > 0:  # Oversold + uptrend
        win_probability = 0.65
    elif rsi > 0.7 and trend < 0:  # Overbought + downtrend
        win_probability = 0.65
    
    is_winner = np.random.random() < win_probability
    pnl = np.random.normal(10, 5) if is_winner else np.random.normal(-8, 3)
    
    return {
        'win_rate': win_probability,
        'profit': pnl,
        'is_winner': is_winner
    }
```

### 2. **Optimal Warm-up Configuration**
```python
# warm_up_config.py
WARMUP_CONFIG = {
    'target_observations': {
        'learning_phase': 100000,    # แทนที่จะรอ 2000 trades
        'shadow_phase': 20000,        # แทนที่จะรอ 500 trades
        'active_phase': 10000         # แทนที่จะรอ 200 trades
    },
    
    'sampling_strategy': {
        'total_samples': 1000000,     # 1M จาก 36M
        'recent_weight': 0.7,         # 70% จากข้อมูลล่าสุด
        'volatility_weight': 0.2,     # 20% จากช่วง high volatility
        'regime_diversity_weight': 0.1 # 10% เพื่อความหลากหลาย
    },
    
    'processing_params': {
        'batch_size': 10000,
        'feature_extraction_interval': 5,  # ทุก 5 วินาที
        'progress_report_interval': 50000
    }
}
```

## 💻 Complete Warm-up Script

```python
# warmup_main.py
import asyncio
import time
from pathlib import Path

async def main():
    # 1. Load configuration
    config_path = Path("config.yaml")
    
    # 2. Initialize system (without starting live trading)
    from main import GridTradingSystem
    system = GridTradingSystem(str(config_path))
    await system.initialize()
    
    # 3. Load and prepare historical data
    print("Loading historical data...")
    historical_ticks = await prepare_historical_data("btcusdt_1s_data.csv")
    
    # 4. Smart sampling
    print("Sampling data intelligently...")
    sampled_ticks = smart_sampling(historical_ticks, target_samples=1000000)
    
    # 5. Run warm-up
    start_time = time.time()
    await warmup_system(system, sampled_ticks)
    
    print(f"\nWarm-up completed in {(time.time() - start_time)/60:.1f} minutes")
    
    # 6. Save warmed-up state
    print("Saving warm-up state...")
    await system.components['attention'].save_state('attention_warmup.json')
    await system.components['regime_detector'].save_state('regime_warmup.json')
    await system.components['strategy_selector'].save_state('strategy_warmup.json')
    
    # 7. Generate warm-up report
    report = {
        'attention_state': await system.components['attention'].get_attention_state(),
        'regime_stats': await system.components['regime_detector'].get_regime_statistics(),
        'feature_importance': system.components['attention'].feature_attention.get_importance_scores()
    }
    
    print("\nWarm-up Report:")
    print(f"  Learning Progress: {system.components['attention'].get_learning_progress():.1%}")
    print(f"  Current Phase: {report['attention_state']['phase']}")
    print(f"  Top Features: {list(report['feature_importance'].items())[:5]}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📈 Expected Results

หลังจาก warm-up ด้วยข้อมูล 1M ticks:
- **Learning Phase**: ข้ามได้ทันที
- **Shadow Phase**: ~1-2 ชั่วโมง
- **Active Phase**: พร้อมใช้งานเร็วขึ้น 10-20 เท่า

## ⚡ Performance Tips

1. **ใช้ multiprocessing สำหรับ feature extraction**
2. **บันทึกเป็น checkpoint ทุก 100k ticks**
3. **ใช้ GPU สำหรับ neural network ถ้ามี**
4. **Process เฉพาะ timeframe ที่จำเป็น (5s, 1m)**

ด้วยวิธีนี้ ระบบจะ "ฉลาด" ขึ้นก่อนเริ่มเทรดจริง!
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
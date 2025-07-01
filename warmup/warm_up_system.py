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
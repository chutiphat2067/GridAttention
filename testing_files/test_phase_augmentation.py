"""
Test Phase-Aware Augmentation Integration
"""

import asyncio
import pytest
import numpy as np
from typing import Dict, List
import time

from attention_learning_layer import AttentionLearningLayer, AttentionPhase
from phase_aware_data_augmenter import (
    PhaseAwareDataAugmenter,
    AugmentationManager,
    create_phase_aware_augmentation_config
)
from market_data_input import MarketTick


class TestPhaseAwareAugmentation:
    """Test phase-aware augmentation integration"""
    
    @pytest.fixture
    async def setup_components(self):
        """Setup test components"""
        # Create attention layer
        attention = AttentionLearningLayer({
            'min_trades_learning': 100,
            'min_trades_shadow': 50,
            'min_trades_active': 25
        })
        
        # Create augmentation manager
        aug_config = create_phase_aware_augmentation_config()
        aug_manager = AugmentationManager(aug_config)
        await aug_manager.initialize(attention)
        
        return attention, aug_manager
        
    async def test_learning_phase_augmentation(self, setup_components):
        """Test augmentation in learning phase"""
        attention, aug_manager = await setup_components
        
        # Verify we're in learning phase
        assert attention.phase == AttentionPhase.LEARNING
        
        # Create test tick
        tick = self._create_test_tick()
        features = self._create_test_features()
        
        # Process with augmentation
        result = await aug_manager.process_tick(
            tick,
            features,
            'ranging',
            {'performance': {'win_rate': 0.5}}
        )
        
        # Should apply augmentation in learning phase
        assert result['augmentation_applied'] is True
        assert result['processed_count'] > 1  # Multiple samples from augmentation
        
        # Check augmentation info
        aug_info = result.get('augmentation_info', {})
        assert aug_info['phase'] == 'learning'
        assert aug_info['augmentation_factor'] >= 2.0  # Should be high in learning
        
    async def test_shadow_phase_augmentation(self, setup_components):
        """Test reduced augmentation in shadow phase"""
        attention, aug_manager = await setup_components
        
        # Force transition to shadow phase
        await self._force_phase_transition(attention, AttentionPhase.SHADOW)
        
        # Process tick
        tick = self._create_test_tick()
        features = self._create_test_features()
        
        result = await aug_manager.process_tick(
            tick,
            features,
            'ranging',
            {'performance': {'win_rate': 0.5}}
        )
        
        # Should apply lighter augmentation
        assert result['augmentation_applied'] is True
        aug_info = result.get('augmentation_info', {})
        assert aug_info['augmentation_factor'] <= 1.5  # Reduced in shadow
        
    async def test_active_phase_no_augmentation(self, setup_components):
        """Test no augmentation in active phase (normal case)"""
        attention, aug_manager = await setup_components
        
        # Force transition to active phase
        await self._force_phase_transition(attention, AttentionPhase.ACTIVE)
        
        # Process with good performance
        tick = self._create_test_tick()
        features = self._create_test_features()
        
        result = await aug_manager.process_tick(
            tick,
            features,
            'ranging',
            {'performance': {'win_rate': 0.55, 'sharpe_ratio': 1.5}}
        )
        
        # Should NOT apply augmentation
        assert result['augmentation_applied'] is False
        assert result['processed_count'] == 1
        
    async def test_active_phase_emergency_augmentation(self, setup_components):
        """Test emergency augmentation when performance drops"""
        attention, aug_manager = await setup_components
        
        # Force transition to active phase
        await self._force_phase_transition(attention, AttentionPhase.ACTIVE)
        
        # Process with poor performance
        tick = self._create_test_tick()
        features = self._create_test_features()
        
        result = await aug_manager.process_tick(
            tick,
            features,
            'ranging',
            {'performance': {'win_rate': 0.40, 'sharpe_ratio': 0.3}}  # Poor performance
        )
        
        # Should apply emergency augmentation
        assert result['augmentation_applied'] is True
        aug_info = result.get('augmentation_info', {})
        assert aug_info['augmentation_factor'] < 1.0  # Very light augmentation
        
    async def test_augmentation_statistics(self, setup_components):
        """Test augmentation statistics tracking"""
        attention, aug_manager = await setup_components
        
        # Process multiple ticks
        for i in range(10):
            tick = self._create_test_tick()
            features = self._create_test_features()
            
            await aug_manager.process_tick(
                tick,
                features,
                'ranging',
                {'performance': {'win_rate': 0.5}}
            )
            
        # Check statistics
        stats = aug_manager.get_stats()
        
        assert stats['total_augmented'] > 10  # Should be more due to augmentation
        assert stats['augmentation_by_phase']['learning'] > 0
        assert stats['current_phase'] == 'learning'
        
    async def test_phase_transition_augmentation_change(self, setup_components):
        """Test augmentation changes with phase transitions"""
        attention, aug_manager = await setup_components
        
        augmentation_factors = []
        
        # Test across all phases
        for phase in [AttentionPhase.LEARNING, AttentionPhase.SHADOW, AttentionPhase.ACTIVE]:
            await self._force_phase_transition(attention, phase)
            
            tick = self._create_test_tick()
            features = self._create_test_features()
            
            result = await aug_manager.process_tick(
                tick,
                features,
                'ranging',
                {'performance': {'win_rate': 0.5}}
            )
            
            if result['augmentation_applied']:
                aug_factor = result['augmentation_info']['augmentation_factor']
                augmentation_factors.append((phase.value, aug_factor))
                
        # Verify decreasing augmentation
        if len(augmentation_factors) >= 2:
            assert augmentation_factors[0][1] > augmentation_factors[1][1]  # Learning > Shadow
            
    def _create_test_tick(self) -> MarketTick:
        """Create test market tick"""
        return MarketTick(
            symbol='BTC/USDT',
            price=50000 + np.random.randn() * 100,
            volume=100 + np.random.exponential(50),
            timestamp=time.time(),
            bid=49995,
            ask=50005,
            exchange='test'
        )
        
    def _create_test_features(self) -> Dict[str, float]:
        """Create test features"""
        return {
            'volatility_5m': 0.001 + np.random.rand() * 0.001,
            'trend_strength': np.random.randn() * 0.5,
            'volume_ratio': 0.8 + np.random.rand() * 0.4,
            'rsi_14': 0.3 + np.random.rand() * 0.4
        }
        
    async def _force_phase_transition(self, attention: AttentionLearningLayer, target_phase: AttentionPhase):
        """Force attention layer to specific phase"""
        attention.phase = target_phase
        attention.metrics.phase = target_phase


async def run_integration_test():
    """Run integration test manually"""
    print("Running Phase-Aware Augmentation Integration Test...")
    
    test = TestPhaseAwareAugmentation()
    
    # Setup
    components = await test.setup_components()
    
    # Run tests
    print("\n1. Testing Learning Phase Augmentation...")
    await test.test_learning_phase_augmentation(components)
    print("✅ Learning phase augmentation working")
    
    print("\n2. Testing Shadow Phase Augmentation...")
    await test.test_shadow_phase_augmentation(components)
    print("✅ Shadow phase augmentation working")
    
    print("\n3. Testing Active Phase (No Augmentation)...")
    await test.test_active_phase_no_augmentation(components)
    print("✅ Active phase normal operation working")
    
    print("\n4. Testing Emergency Augmentation...")
    await test.test_active_phase_emergency_augmentation(components)
    print("✅ Emergency augmentation working")
    
    print("\n5. Testing Statistics...")
    await test.test_augmentation_statistics(components)
    print("✅ Statistics tracking working")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(run_integration_test())
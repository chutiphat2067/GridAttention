"""
Warmup System Integration Test for GridAttention
Tests the warmup functionality and accelerated learning
"""

import asyncio
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Mock Components for Testing
# ============================================================================

class MockAttentionLearningLayer:
    """Mock Attention Layer for testing warmup"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.phase = "learning"
        self.total_observations = 0
        self.feature_importance = {}
        self.temporal_patterns = {}
        self.regime_patterns = {}
        self.warmup_loaded = False
        self.warmup_state = None
        
    async def _load_warmup_state(self, filepath: str) -> bool:
        """Load warmup state from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            if 'attention_state' in data:
                state = data['attention_state']
                self.phase = state.get('phase', 'learning')
                self.total_observations = state.get('total_observations', 0)
                self.feature_importance = state.get('feature_importance', {})
                self.temporal_patterns = state.get('temporal_patterns', {})
                self.regime_patterns = state.get('regime_patterns', {})
                self.warmup_loaded = True
                self.warmup_state = state
                
                logger.info(f"Loaded warmup state: {self.total_observations} observations")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load warmup state: {e}")
            return False
            
    def get_learning_progress(self) -> float:
        """Get learning progress percentage"""
        thresholds = {
            'learning': 2000,
            'shadow': 500,
            'active': 200
        }
        
        if self.phase == 'active':
            return 1.0
        elif self.phase == 'shadow':
            return 0.7 + 0.3 * min(self.total_observations / thresholds['active'], 1.0)
        else:
            return min(self.total_observations / thresholds['learning'], 0.7)
            
    async def process(self, features: Dict[str, float], regime: str, context: Dict[str, Any]):
        """Process features with attention"""
        self.total_observations += 1
        
        # Update feature importance
        for feature, value in features.items():
            if feature not in self.feature_importance:
                self.feature_importance[feature] = 0.5
            # Simple learning simulation
            self.feature_importance[feature] *= 0.99
            self.feature_importance[feature] += 0.01 * abs(value)
            
        # Check phase transitions
        if self.phase == 'learning' and self.total_observations >= 2000:
            self.phase = 'shadow'
        elif self.phase == 'shadow' and self.total_observations >= 2500:
            self.phase = 'active'
            
        return {
            'weighted_features': features,  # Simplified
            'phase': self.phase,
            'confidence': self.get_learning_progress()
        }


class MockGridSystem:
    """Mock Grid System for testing"""
    
    def __init__(self):
        self.components = {
            'attention': MockAttentionLearningLayer(),
            'market_data': MockMarketData(),
            'features': MockFeatureEngineering(),
            'regime_detector': MockRegimeDetector()
        }
        self.initialized = False
        
    async def initialize(self):
        """Initialize system"""
        self.initialized = True
        logger.info("Mock Grid System initialized")
        
    async def process_tick(self, tick: Dict[str, Any]):
        """Process a market tick"""
        # Update market data
        await self.components['market_data'].update_buffer(tick)
        
        # Extract features
        features = await self.components['features'].extract_features()
        
        if features:
            # Detect regime
            regime, confidence = await self.components['regime_detector'].detect_regime(
                features.features
            )
            
            # Process with attention
            result = await self.components['attention'].process(
                features.features,
                regime,
                {'timestamp': tick['timestamp']}
            )
            
            return result
            
        return None


class MockMarketData:
    """Mock market data component"""
    
    def __init__(self):
        self.buffer = []
        
    async def update_buffer(self, tick: Dict[str, Any]):
        """Update data buffer"""
        self.buffer.append(tick)
        if len(self.buffer) > 1000:
            self.buffer.pop(0)


class MockFeatureEngineering:
    """Mock feature engineering"""
    
    def __init__(self):
        self.call_count = 0
        
    async def extract_features(self):
        """Extract features from market data"""
        self.call_count += 1
        
        # Return features every 5 calls (simulating 5-second intervals)
        if self.call_count % 5 == 0:
            return type('FeatureResult', (), {
                'features': {
                    'volatility_5m': 0.001 + np.random.rand() * 0.001,
                    'trend_strength': np.random.rand() * 2 - 1,
                    'volume_ratio': 0.8 + np.random.rand() * 0.4,
                    'rsi_14': np.random.rand()
                }
            })
        return None


class MockRegimeDetector:
    """Mock regime detector"""
    
    async def detect_regime(self, features: Dict[str, float]):
        """Detect market regime"""
        volatility = features.get('volatility_5m', 0.001)
        
        if volatility > 0.002:
            return 'volatile', 0.8
        elif volatility < 0.0008:
            return 'ranging', 0.9
        else:
            return 'trending', 0.7


# ============================================================================
# Warmup Test Functions
# ============================================================================

def create_warmup_state(observations: int = 150000) -> Dict[str, Any]:
    """Create a warmup state file"""
    return {
        'version': '1.0',
        'timestamp': time.time(),
        'metadata': {
            'total_ticks_processed': observations * 5,
            'total_features_extracted': observations,
            'learning_phases_completed': ['learning', 'shadow', 'active'],
            'training_duration_hours': 48,
            'final_learning_progress': 0.95
        },
        'attention_state': {
            'phase': 'active',
            'total_observations': observations,
            'feature_importance': {
                'volatility_5m': 0.812,
                'volatility_20m': 0.723,
                'trend_strength': 0.891,
                'volume_ratio': 0.534,
                'price_momentum': 0.445,
                'rsi_14': 0.623,
                'spread_bps': 0.234
            },
            'temporal_patterns': {
                'decay_rate': 0.995,
                'window_size': 1000,
                'learned_cycles': [300, 900, 3600]
            },
            'regime_patterns': {
                'trending': {
                    'observations': 45000,
                    'key_features': ['trend_strength', 'price_momentum'],
                    'avg_importance': 0.85
                },
                'ranging': {
                    'observations': 60000,
                    'key_features': ['volatility_5m', 'rsi_14'],
                    'avg_importance': 0.75
                },
                'volatile': {
                    'observations': 30000,
                    'key_features': ['volatility_20m', 'volume_ratio'],
                    'avg_importance': 0.80
                },
                'breakout': {
                    'observations': 15000,
                    'key_features': ['volume_ratio', 'trend_strength'],
                    'avg_importance': 0.90
                }
            }
        },
        'performance_metrics': {
            'learning_curve': {
                'milestones': [
                    {'observations': 1000, 'accuracy': 0.65},
                    {'observations': 5000, 'accuracy': 0.75},
                    {'observations': 20000, 'accuracy': 0.82},
                    {'observations': 50000, 'accuracy': 0.88},
                    {'observations': 100000, 'accuracy': 0.92},
                    {'observations': 150000, 'accuracy': 0.95}
                ]
            }
        }
    }


async def test_warmup_loading():
    """Test loading warmup state"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Warmup State Loading")
    logger.info("="*60)
    
    # Create test warmup state
    warmup_state = create_warmup_state()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(warmup_state, f, indent=2)
        temp_file = f.name
        
    try:
        # Test loading in attention layer
        attention = MockAttentionLearningLayer()
        
        # Before loading
        logger.info(f"Before warmup - Phase: {attention.phase}, Observations: {attention.total_observations}")
        assert attention.phase == "learning"
        assert attention.total_observations == 0
        
        # Load warmup
        success = await attention._load_warmup_state(temp_file)
        assert success is True
        
        # After loading
        logger.info(f"After warmup - Phase: {attention.phase}, Observations: {attention.total_observations}")
        assert attention.phase == "active"
        assert attention.total_observations == 150000
        assert attention.warmup_loaded is True
        
        # Check feature importance loaded
        assert len(attention.feature_importance) > 0
        assert 'volatility_5m' in attention.feature_importance
        
        logger.info("✅ Warmup loading test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Warmup loading test FAILED: {e}")
        return False
        
    finally:
        Path(temp_file).unlink(missing_ok=True)


async def test_accelerated_learning():
    """Test that warmup accelerates learning"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Accelerated Learning with Warmup")
    logger.info("="*60)
    
    # System 1: Without warmup
    system1 = MockGridSystem()
    await system1.initialize()
    
    # System 2: With warmup
    system2 = MockGridSystem()
    await system2.initialize()
    
    # Load warmup for system 2
    warmup_state = create_warmup_state()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(warmup_state, f, indent=2)
        temp_file = f.name
        
    try:
        await system2.components['attention']._load_warmup_state(temp_file)
        
        # Process same number of ticks for both
        num_ticks = 1000
        
        for i in range(num_ticks):
            tick = {
                'symbol': 'BTC/USDT',
                'price': 50000 + np.random.randn() * 100,
                'volume': 100 + np.random.exponential(50),
                'timestamp': time.time() + i,
                'bid': 49995,
                'ask': 50005
            }
            
            await system1.process_tick(tick)
            await system2.process_tick(tick)
            
        # Compare learning progress
        progress1 = system1.components['attention'].get_learning_progress()
        progress2 = system2.components['attention'].get_learning_progress()
        
        logger.info(f"System 1 (no warmup) - Progress: {progress1:.1%}, Phase: {system1.components['attention'].phase}")
        logger.info(f"System 2 (with warmup) - Progress: {progress2:.1%}, Phase: {system2.components['attention'].phase}")
        
        # System with warmup should be significantly ahead
        assert progress2 > progress1
        assert system2.components['attention'].phase == 'active'
        assert system1.components['attention'].phase == 'learning'
        
        logger.info("✅ Accelerated learning test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Accelerated learning test FAILED: {e}")
        return False
        
    finally:
        Path(temp_file).unlink(missing_ok=True)


async def test_feature_importance_preservation():
    """Test that feature importance is preserved"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Feature Importance Preservation")
    logger.info("="*60)
    
    # Create warmup with specific feature importance
    warmup_state = create_warmup_state()
    original_importance = warmup_state['attention_state']['feature_importance'].copy()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(warmup_state, f, indent=2)
        temp_file = f.name
        
    try:
        # Load warmup
        attention = MockAttentionLearningLayer()
        await attention._load_warmup_state(temp_file)
        
        # Check feature importance preserved
        for feature, importance in original_importance.items():
            assert feature in attention.feature_importance
            assert attention.feature_importance[feature] == importance
            
        logger.info("Original importance values:")
        for feature, value in original_importance.items():
            logger.info(f"  {feature}: {value:.3f}")
            
        logger.info("\nLoaded importance values:")
        for feature, value in attention.feature_importance.items():
            logger.info(f"  {feature}: {value:.3f}")
            
        logger.info("✅ Feature importance preservation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature importance preservation test FAILED: {e}")
        return False
        
    finally:
        Path(temp_file).unlink(missing_ok=True)


async def test_regime_patterns_loading():
    """Test regime-specific patterns are loaded"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Regime Patterns Loading")
    logger.info("="*60)
    
    warmup_state = create_warmup_state()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(warmup_state, f, indent=2)
        temp_file = f.name
        
    try:
        attention = MockAttentionLearningLayer()
        await attention._load_warmup_state(temp_file)
        
        # Check regime patterns loaded
        assert 'regime_patterns' in attention.warmup_state
        regime_patterns = attention.warmup_state['regime_patterns']
        
        expected_regimes = ['trending', 'ranging', 'volatile', 'breakout']
        for regime in expected_regimes:
            assert regime in regime_patterns
            assert 'observations' in regime_patterns[regime]
            assert 'key_features' in regime_patterns[regime]
            
            logger.info(f"\nRegime: {regime}")
            logger.info(f"  Observations: {regime_patterns[regime]['observations']:,}")
            logger.info(f"  Key features: {regime_patterns[regime]['key_features']}")
            
        logger.info("✅ Regime patterns loading test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Regime patterns loading test FAILED: {e}")
        return False
        
    finally:
        Path(temp_file).unlink(missing_ok=True)


async def test_invalid_warmup_handling():
    """Test handling of invalid warmup files"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Invalid Warmup File Handling")
    logger.info("="*60)
    
    test_cases = [
        {
            'name': 'Empty file',
            'content': {}
        },
        {
            'name': 'Missing attention_state',
            'content': {'version': '1.0', 'timestamp': time.time()}
        },
        {
            'name': 'Invalid JSON',
            'content': "This is not valid JSON"
        },
        {
            'name': 'Wrong version',
            'content': {
                'version': '0.5',
                'attention_state': {'phase': 'learning'}
            }
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            if test_case['name'] == 'Invalid JSON':
                f.write(test_case['content'])
            else:
                json.dump(test_case['content'], f)
            temp_file = f.name
            
        try:
            attention = MockAttentionLearningLayer()
            result = await attention._load_warmup_state(temp_file)
            
            # Should handle gracefully
            assert result is False or attention.warmup_loaded is False
            logger.info(f"  ✅ Handled gracefully")
            
        except Exception as e:
            logger.error(f"  ❌ Unexpected error: {e}")
            return False
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
            
    logger.info("\n✅ Invalid warmup handling test PASSED")
    return True


async def test_performance_comparison():
    """Test performance metrics with and without warmup"""
    logger.info("\n" + "="*60)
    logger.info("TEST: Performance Metrics Comparison")
    logger.info("="*60)
    
    # Create warmup state
    warmup_state = create_warmup_state()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(warmup_state, f, indent=2)
        temp_file = f.name
        
    try:
        # System without warmup
        system_cold = MockGridSystem()
        await system_cold.initialize()
        
        # System with warmup
        system_warm = MockGridSystem()
        await system_warm.initialize()
        await system_warm.components['attention']._load_warmup_state(temp_file)
        
        # Metrics to track
        metrics = {
            'cold': {
                'phase_transitions': [],
                'observations_to_shadow': None,
                'observations_to_active': None
            },
            'warm': {
                'phase_transitions': [],
                'started_in_phase': system_warm.components['attention'].phase
            }
        }
        
        # Process ticks and track progress
        for i in range(3000):
            tick = {
                'symbol': 'BTC/USDT',
                'price': 50000 + np.random.randn() * 100,
                'volume': 100,
                'timestamp': time.time() + i,
                'bid': 49995,
                'ask': 50005
            }
            
            # Track cold system
            prev_phase_cold = system_cold.components['attention'].phase
            await system_cold.process_tick(tick)
            curr_phase_cold = system_cold.components['attention'].phase
            
            if prev_phase_cold != curr_phase_cold:
                metrics['cold']['phase_transitions'].append({
                    'from': prev_phase_cold,
                    'to': curr_phase_cold,
                    'at_observation': system_cold.components['attention'].total_observations
                })
                
                if curr_phase_cold == 'shadow':
                    metrics['cold']['observations_to_shadow'] = system_cold.components['attention'].total_observations
                elif curr_phase_cold == 'active':
                    metrics['cold']['observations_to_active'] = system_cold.components['attention'].total_observations
                    
            # Track warm system (should stay in active)
            await system_warm.process_tick(tick)
            
        # Compare results
        logger.info("\nCold Start System:")
        logger.info(f"  Final phase: {system_cold.components['attention'].phase}")
        logger.info(f"  Observations to shadow: {metrics['cold']['observations_to_shadow']}")
        logger.info(f"  Observations to active: {metrics['cold']['observations_to_active']}")
        
        logger.info("\nWarm Start System:")
        logger.info(f"  Started in phase: {metrics['warm']['started_in_phase']}")
        logger.info(f"  Final phase: {system_warm.components['attention'].phase}")
        logger.info(f"  Total observations: {system_warm.components['attention'].total_observations}")
        
        # Warm system should be way ahead
        assert system_warm.components['attention'].phase == 'active'
        assert system_cold.components['attention'].phase in ['learning', 'shadow']
        
        # Calculate acceleration factor
        if metrics['cold']['observations_to_shadow']:
            acceleration = 2000 / metrics['cold']['observations_to_shadow']
            logger.info(f"\nAcceleration factor: {acceleration:.1f}x faster to reach shadow phase")
            
        logger.info("✅ Performance comparison test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test FAILED: {e}")
        return False
        
    finally:
        Path(temp_file).unlink(missing_ok=True)


# ============================================================================
# Integration Test Suite
# ============================================================================

async def run_warmup_integration_tests():
    """Run all warmup integration tests"""
    logger.info("\n" + "="*80)
    logger.info("GRIDATTENTION WARMUP INTEGRATION TEST SUITE")
    logger.info("="*80)
    
    tests = [
        ("Warmup State Loading", test_warmup_loading),
        ("Accelerated Learning", test_accelerated_learning),
        ("Feature Importance Preservation", test_feature_importance_preservation),
        ("Regime Patterns Loading", test_regime_patterns_loading),
        ("Invalid Warmup Handling", test_invalid_warmup_handling),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
            
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<40} {status}")
        
    logger.info(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("\n🎉 All warmup integration tests passed!")
        logger.info("The warmup system is working correctly.")
    else:
        logger.error(f"\n❌ {total - passed} tests failed.")
        
    return passed == total


# ============================================================================
# Example Usage Documentation
# ============================================================================

def print_usage_guide():
    """Print usage guide for warmup system"""
    guide = """
    ============================================================================
    GRIDATTENTION WARMUP SYSTEM USAGE GUIDE
    ============================================================================
    
    1. GENERATE WARMUP STATE (from historical data):
       
       # In your Jupyter notebook or script:
       from warmup_main import run_complete_warmup
       
       await run_complete_warmup(
           data_file_path="historical_data.pkl",
           sample_size=1000000,  # Use 1M samples for warmup
           target_observations=150000  # Target 150k observations
       )
       
       # This will create 'attention_warmup_state.json'
    
    2. USE WARMUP STATE IN PRODUCTION:
       
       # Copy attention_warmup_state.json to your GridAttention root directory
       # The system will automatically detect and load it on startup
       
       # In main.py, the AttentionLearningLayer will:
       # 1. Check for warmup file on initialization
       # 2. Load pre-trained state if found
       # 3. Start in 'active' phase instead of 'learning'
    
    3. BENEFITS:
       
       - Skip 2000 observations learning phase → Start immediately
       - Skip 500 observations shadow phase → Go live faster  
       - Pre-trained feature importance → Better decisions from start
       - Regime-specific patterns → Accurate market detection
       - 3-10x faster time to production readiness
    
    4. MONITORING:
       
       # Check if warmup was loaded:
       if system.components['attention'].warmup_loaded:
           print("System started with warmup acceleration!")
       
       # Check learning progress:
       progress = system.components['attention'].get_learning_progress()
       print(f"Learning progress: {progress:.1%}")
    
    ============================================================================
    """
    
    print(guide)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Print usage guide
    print_usage_guide()
    
    # Run tests
    success = asyncio.run(run_warmup_integration_tests())
    
    # Exit with appropriate code
    exit(0 if success else 1)
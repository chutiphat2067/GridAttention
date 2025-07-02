# tests/test_overfitting_detection.py
"""
Comprehensive testing suite for overfitting detection system
Tests detection accuracy, recovery mechanisms, and integration

Author: Grid Trading System
Date: 2024
"""

import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time
from dataclasses import dataclass

# Import modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overfitting_detector import (
    OverfittingDetector, 
    OverfittingType,
    OverfittingSeverity,
    OverfittingMetrics,
    PerformanceDivergenceDetector,
    ConfidenceCalibrationChecker,
    OverfittingRecovery
)


class TestOverfittingDetection(unittest.TestCase):
    """Test overfitting detection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = OverfittingDetector({
            'performance_gap_threshold': 0.15,
            'confidence_threshold': 0.2,
            'feature_stability_threshold': 0.3
        })
        
    def test_performance_divergence_detection(self):
        """Test detection of train/test performance divergence"""
        # Simulate overfitted performance
        for i in range(100):
            # Training performs well
            self.detector.performance_detector.add_training_metric({
                'win_rate': 0.65 + np.random.normal(0, 0.02),
                'profit_factor': 1.5 + np.random.normal(0, 0.05)
            })
            
            # Live performance is worse
            self.detector.performance_detector.add_live_metric({
                'win_rate': 0.48 + np.random.normal(0, 0.02),
                'profit_factor': 1.1 + np.random.normal(0, 0.05)
            })
        
        # Calculate divergence
        divergence, details = self.detector.performance_detector.calculate_divergence()
        
        # Assert overfitting detected
        self.assertGreater(divergence, 0.15)
        self.assertIn('win_rate_gap', details)
        self.assertGreater(details['win_rate_gap'], 0.15)
        
    def test_confidence_calibration(self):
        """Test confidence calibration checking"""
        checker = ConfidenceCalibrationChecker()
        
        # Add miscalibrated predictions
        for _ in range(100):
            # Model is 90% confident but only 60% accurate
            confidence = 0.9
            actual_correct = np.random.random() < 0.6
            checker.add_prediction(confidence, actual_correct)
            
            # Model is 60% confident but 80% accurate
            confidence = 0.6
            actual_correct = np.random.random() < 0.8
            checker.add_prediction(confidence, actual_correct)
        
        # Calculate calibration error
        error, calibration_map = checker.calculate_calibration_error()
        
        # Should detect miscalibration
        self.assertGreater(error, 0.2)
        
    def test_feature_stability_monitoring(self):
        """Test feature importance stability tracking"""
        # Simulate unstable feature importance
        for i in range(50):
            if i < 25:
                # Stable period
                importance = {
                    'feature1': 0.3 + np.random.normal(0, 0.02),
                    'feature2': 0.5 + np.random.normal(0, 0.02),
                    'feature3': 0.2 + np.random.normal(0, 0.02)
                }
            else:
                # Unstable period - feature importance changes drastically
                importance = {
                    'feature1': 0.6 + np.random.normal(0, 0.02),
                    'feature2': 0.1 + np.random.normal(0, 0.02),
                    'feature3': 0.3 + np.random.normal(0, 0.02)
                }
                
            self.detector.feature_monitor.update_importance(importance)
        
        # Calculate stability
        stability_score, unstable_features = self.detector.feature_monitor.calculate_stability()
        
        # Should detect instability
        self.assertLess(stability_score, 0.7)
        self.assertIn('feature1', unstable_features)
        self.assertIn('feature2', unstable_features)
        
    async def test_comprehensive_detection(self):
        """Test complete overfitting detection flow"""
        # Set up overfitting scenario
        await self._simulate_overfitting_scenario()
        
        # Detect overfitting
        detection = await self.detector.detect_overfitting()
        
        # Verify detection
        self.assertTrue(detection['is_overfitting'])
        self.assertIn('PERFORMANCE_DIVERGENCE', detection['overfitting_types'])
        self.assertEqual(detection['severity'], 'HIGH')
        self.assertGreater(len(detection['recommendations']), 0)
        
    async def test_severity_calculation(self):
        """Test overfitting severity calculation"""
        # Test different scenarios
        scenarios = [
            {
                'performance_gap': 0.05,
                'calibration_error': 0.1,
                'feature_stability': 0.9,
                'expected_severity': OverfittingSeverity.LOW
            },
            {
                'performance_gap': 0.25,
                'calibration_error': 0.3,
                'feature_stability': 0.5,
                'expected_severity': OverfittingSeverity.HIGH
            },
            {
                'performance_gap': 0.4,
                'calibration_error': 0.4,
                'feature_stability': 0.3,
                'expected_severity': OverfittingSeverity.CRITICAL
            }
        ]
        
        for scenario in scenarios:
            metrics = OverfittingMetrics(
                performance_gap=scenario['performance_gap'],
                confidence_calibration_error=scenario['calibration_error'],
                feature_stability_score=scenario['feature_stability']
            )
            
            severity = metrics.get_severity()
            self.assertEqual(severity, scenario['expected_severity'])
            
    async def _simulate_overfitting_scenario(self):
        """Helper to simulate overfitting conditions"""
        # Add divergent performance
        for _ in range(100):
            await self.detector.add_training_result(
                win_rate=0.7,
                profit_factor=1.6
            )
            await self.detector.add_live_result(
                win_rate=0.45,
                profit_factor=0.95
            )
            
        # Add miscalibrated predictions
        for _ in range(50):
            await self.detector.add_prediction(
                confidence=0.85,
                actual=np.random.random() < 0.5
            )
            
        # Add unstable features
        await self.detector.update_feature_importance({
            'feature1': 0.8,
            'feature2': 0.1,
            'feature3': 0.1
        })


class TestOverfittingRecovery(unittest.TestCase):
    """Test overfitting recovery mechanisms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_components = {
            'risk_manager': Mock(risk_reduction_mode=False),
            'grid_strategy_selector': Mock(),
            'attention': Mock(),
            'feedback_loop': Mock(learning_rate=0.001)
        }
        
        self.recovery = OverfittingRecovery(self.mock_components)
        
    async def test_critical_recovery(self):
        """Test critical overfitting recovery"""
        detection = {
            'severity': 'CRITICAL',
            'overfitting_types': ['PERFORMANCE_DIVERGENCE', 'CONFIDENCE_MISCALIBRATION']
        }
        
        result = await self.recovery.recover_from_overfitting(
            detection,
            OverfittingSeverity.CRITICAL
        )
        
        # Verify actions taken
        self.assertTrue(result['success'])
        self.assertIn('Enabled risk reduction mode', result['actions_taken'])
        self.assertIn('Reduced position sizes by 70%', result['actions_taken'])
        
        # Verify component changes
        self.assertTrue(self.mock_components['risk_manager'].risk_reduction_mode)
        
    async def test_high_severity_recovery(self):
        """Test high severity recovery"""
        detection = {
            'severity': 'HIGH',
            'overfitting_types': ['FEATURE_INSTABILITY']
        }
        
        result = await self.recovery.recover_from_overfitting(
            detection,
            OverfittingSeverity.HIGH
        )
        
        # Verify gradual adjustment
        self.assertTrue(result['success'])
        self.assertIn('Reduced learning rate by 50%', result['actions_taken'])
        self.assertEqual(self.mock_components['feedback_loop'].learning_rate, 0.0005)
        
    async def test_recovery_failure_handling(self):
        """Test recovery failure scenarios"""
        # Simulate component failure
        self.mock_components['risk_manager'].side_effect = Exception("Component failure")
        
        detection = {'severity': 'CRITICAL'}
        
        result = await self.recovery.recover_from_overfitting(
            detection,
            OverfittingSeverity.CRITICAL
        )
        
        # Should handle failure gracefully
        self.assertFalse(result['success'])
        self.assertIn('error', result)


class TestIntegration(unittest.TestCase):
    """Integration tests for overfitting detection system"""
    
    async def test_end_to_end_detection_and_recovery(self):
        """Test complete detection and recovery flow"""
        # Initialize system
        detector = OverfittingDetector()
        components = self._create_mock_components()
        recovery = OverfittingRecovery(components)
        
        # Simulate trading with overfitting
        await self._simulate_overfitted_trading(detector)
        
        # Detect overfitting
        detection = await detector.detect_overfitting()
        
        # Should detect overfitting
        self.assertTrue(detection['is_overfitting'])
        
        # Execute recovery
        severity = OverfittingSeverity[detection['severity']]
        recovery_result = await recovery.recover_from_overfitting(
            detection,
            severity
        )
        
        # Verify recovery executed
        self.assertTrue(recovery_result['success'])
        self.assertGreater(len(recovery_result['actions_taken']), 0)
        
    async def test_monitoring_integration(self):
        """Test integration with monitoring system"""
        from overfitting_detector import OverfittingMonitor
        
        detector = OverfittingDetector()
        monitor = OverfittingMonitor(detector)
        
        # Track alerts
        alerts_received = []
        
        async def alert_handler(alert):
            alerts_received.append(alert)
            
        monitor.register_alert_handler(alert_handler)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Simulate overfitting
        await self._simulate_overfitted_trading(detector)
        
        # Wait for monitoring cycle
        await asyncio.sleep(0.1)
        
        # Should have received alerts
        self.assertGreater(len(alerts_received), 0)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
    def _create_mock_components(self):
        """Create mock components for testing"""
        return {
            'risk_manager': Mock(risk_reduction_mode=False),
            'grid_strategy_selector': Mock(strategies={'symmetric': Mock()}),
            'attention': AsyncMock(),
            'feedback_loop': Mock(learning_rate=0.001)
        }
        
    async def _simulate_overfitted_trading(self, detector):
        """Simulate trading with overfitting characteristics"""
        # Good training performance
        for _ in range(50):
            await detector.add_training_result(
                win_rate=0.68,
                profit_factor=1.55
            )
            
        # Poor live performance
        for _ in range(50):
            await detector.add_live_result(
                win_rate=0.42,
                profit_factor=0.88
            )
            
        # Miscalibrated predictions
        for _ in range(30):
            await detector.add_prediction(
                confidence=0.8,
                actual=False
            )


class TestPerformance(unittest.TestCase):
    """Performance tests for overfitting detection"""
    
    async def test_detection_latency(self):
        """Test detection speed"""
        detector = OverfittingDetector()
        
        # Add minimal data
        for _ in range(100):
            await detector.add_training_result(0.6, 1.3)
            await detector.add_live_result(0.5, 1.1)
            
        # Measure detection time
        start = time.time()
        detection = await detector.detect_overfitting()
        duration = time.time() - start
        
        # Should be fast
        self.assertLess(duration, 0.1)  # Less than 100ms
        
    async def test_memory_usage(self):
        """Test memory efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        detector = OverfittingDetector()
        
        # Add lots of data
        for _ in range(10000):
            await detector.add_training_result(
                np.random.random(),
                np.random.random() * 2
            )
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should use reasonable memory
        self.assertLess(memory_increase, 100)  # Less than 100MB increase


# Test utilities
def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
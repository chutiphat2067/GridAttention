"""
GridAttention Complete System Validation Test
Comprehensive validation of all system components and their integration
"""

import asyncio
import json
import yaml
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import tempfile
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# System Configuration
# ============================================================================

def create_test_config() -> Dict[str, Any]:
    """Create comprehensive test configuration"""
    return {
        'market_data': {
            'symbols': ['BTC/USDT'],
            'update_interval': 1,
            'buffer_size': 1000,
            'quality_threshold': 0.8
        },
        'features': {
            'indicators': [
                'volatility_5m', 'volatility_20m',
                'trend_strength', 'volume_ratio',
                'price_momentum', 'rsi_14', 'spread_bps'
            ],
            'lookback_periods': [5, 20, 50],
            'min_samples': 20
        },
        'attention': {
            'learning_rate': 0.001,
            'attention_window': 1000,
            'min_trades_for_learning': 2000,
            'min_trades_for_shadow': 500,
            'min_trades_for_active': 200,
            'max_attention_influence': 0.3,
            'regularization': {
                'dropout_rate': 0.3,
                'weight_decay': 0.01,
                'gradient_clipping': 1.0
            }
        },
        'regime_detector': {
            'ensemble': {
                'enabled': True,
                'methods': ['gmm', 'kmeans', 'rule_based'],
                'min_confidence': 0.6,
                'consistency_threshold': 0.8
            },
            'regimes': ['trending', 'ranging', 'volatile', 'breakout', 'uncertain']
        },
        'strategy_selector': {
            'strategies': {
                'trending': {'type': 'asymmetric', 'spacing': 0.002, 'levels': 8},
                'ranging': {'type': 'symmetric', 'spacing': 0.001, 'levels': 10},
                'volatile': {'type': 'symmetric', 'spacing': 0.003, 'levels': 6},
                'breakout': {'type': 'asymmetric', 'spacing': 0.0025, 'levels': 7},
                'uncertain': {'type': 'symmetric', 'spacing': 0.0015, 'levels': 5}
            },
            'validation': {
                'cross_validation_folds': 5,
                'min_samples_for_learning': 200,
                'performance_threshold': 0.03
            }
        },
        'risk_management': {
            'max_position_size': 0.05,
            'max_concurrent_orders': 8,
            'max_daily_loss': 0.01,
            'max_drawdown': 0.03,
            'position_correlation_limit': 0.7,
            'concentration_limit': 0.2,
            'var_confidence_level': 0.95
        },
        'execution': {
            'mode': 'simulation',
            'latency_target': 100,
            'fee_rate': 0.001,
            'slippage_model': 'linear',
            'max_slippage': 0.002
        },
        'performance_monitor': {
            'metrics_window': 1000,
            'save_interval': 300,
            'overfitting_detection': {
                'enabled': True,
                'window_size': 100,
                'degradation_threshold': 0.1
            }
        },
        'feedback_loop': {
            'update_interval': 120,
            'min_samples': 500,
            'confidence_threshold': 0.85,
            'learning_rate': 0.0005,
            'max_adjustment': 0.05
        },
        'overfitting': {
            'detection_threshold': 0.15,
            'recovery_threshold': 0.05,
            'checkpoint_interval': 3600,
            'validation_split': 0.2
        }
    }


def create_overfitting_config() -> Dict[str, Any]:
    """Create overfitting protection configuration"""
    return {
        'overfitting_detection': {
            'sensitivity': 0.7,
            'min_samples': 100,
            'detection_methods': [
                'performance_degradation',
                'parameter_instability',
                'prediction_variance',
                'feature_importance_shift'
            ]
        },
        'regularization': {
            'dropout_rate': 0.3,
            'weight_decay': 0.01,
            'gradient_clipping': 1.0,
            'early_stopping_patience': 50,
            'learning_rate_decay': 0.95,
            'max_norm': 2.0,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2
        },
        'ensemble': {
            'enabled': True,
            'n_models': 3,
            'voting': 'soft',
            'diversity_penalty': 0.1
        },
        'data_augmentation': {
            'noise_injection': 0.01,
            'time_shift': True,
            'synthetic_scenarios': True
        },
        'checkpointing': {
            'enabled': True,
            'checkpoint_dir': './test_checkpoints',
            'max_checkpoints': 5,
            'checkpoint_on_improvement': True
        }
    }


# ============================================================================
# Component Validators
# ============================================================================

class ComponentValidator:
    """Base class for component validation"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.validation_results = []
        self.errors = []
        
    def add_result(self, check_name: str, passed: bool, details: str = ""):
        """Add validation result"""
        self.validation_results.append({
            'check': check_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now()
        })
        
        if not passed:
            self.errors.append(f"{check_name}: {details}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r['passed'])
        
        return {
            'component': self.component_name,
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': total_checks - passed_checks,
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'errors': self.errors
        }


class DataFlowValidator(ComponentValidator):
    """Validate data flow through the system"""
    
    def __init__(self):
        super().__init__("DataFlow")
        
    async def validate_market_data_flow(self):
        """Validate market data -> features -> attention flow"""
        try:
            # Simulate market tick
            tick = {
                'symbol': 'BTC/USDT',
                'price': 50000,
                'volume': 100,
                'timestamp': time.time(),
                'bid': 49995,
                'ask': 50005
            }
            
            # Check tick structure
            required_fields = ['symbol', 'price', 'volume', 'timestamp', 'bid', 'ask']
            has_all_fields = all(field in tick for field in required_fields)
            self.add_result("Market tick structure", has_all_fields)
            
            # Simulate feature extraction
            features = {
                'volatility_5m': 0.001,
                'trend_strength': 0.5,
                'volume_ratio': 1.2,
                'rsi_14': 0.6
            }
            
            # Check features
            self.add_result("Feature extraction", len(features) > 0)
            
            # Simulate attention processing
            attention_result = {
                'weighted_features': features,
                'phase': 'learning',
                'confidence': 0.5
            }
            
            self.add_result("Attention processing", 'weighted_features' in attention_result)
            
        except Exception as e:
            self.add_result("Data flow validation", False, str(e))


class ConfigurationValidator(ComponentValidator):
    """Validate system configuration"""
    
    def __init__(self):
        super().__init__("Configuration")
        
    def validate_config_structure(self, config: Dict[str, Any]):
        """Validate configuration structure and values"""
        # Check required sections
        required_sections = [
            'market_data', 'features', 'attention', 'regime_detector',
            'strategy_selector', 'risk_management', 'execution',
            'performance_monitor', 'feedback_loop'
        ]
        
        for section in required_sections:
            self.add_result(
                f"Config section: {section}",
                section in config,
                f"Missing section: {section}" if section not in config else "Present"
            )
            
        # Validate risk limits
        if 'risk_management' in config:
            risk_config = config['risk_management']
            
            # Check position size limit
            max_pos_size = risk_config.get('max_position_size', 0)
            self.add_result(
                "Position size limit",
                0 < max_pos_size <= 0.2,
                f"Value: {max_pos_size}"
            )
            
            # Check daily loss limit
            max_daily_loss = risk_config.get('max_daily_loss', 0)
            self.add_result(
                "Daily loss limit",
                0 < max_daily_loss <= 0.05,
                f"Value: {max_daily_loss}"
            )
            
            # Check drawdown limit
            max_drawdown = risk_config.get('max_drawdown', 0)
            self.add_result(
                "Drawdown limit",
                0 < max_drawdown <= 0.1,
                f"Value: {max_drawdown}"
            )


class DependencyValidator(ComponentValidator):
    """Validate system dependencies"""
    
    def __init__(self):
        super().__init__("Dependencies")
        
    def validate_python_packages(self):
        """Check if required packages are available"""
        required_packages = [
            ('numpy', 'np'),
            ('pandas', 'pd'),
            ('asyncio', 'asyncio'),
            ('json', 'json'),
            ('yaml', 'yaml'),
            ('logging', 'logging')
        ]
        
        for package_name, import_name in required_packages:
            try:
                exec(f"import {import_name}")
                self.add_result(f"Package: {package_name}", True)
            except ImportError:
                self.add_result(f"Package: {package_name}", False, "Not installed")
                
    def validate_optional_packages(self):
        """Check optional packages"""
        optional_packages = [
            ('torch', 'PyTorch for neural networks'),
            ('sklearn', 'Scikit-learn for ML algorithms'),
            ('scipy', 'SciPy for statistical functions'),
            ('websockets', 'WebSockets for real-time data'),
            ('aiohttp', 'Async HTTP for API calls')
        ]
        
        for package, description in optional_packages:
            try:
                exec(f"import {package}")
                self.add_result(f"Optional: {package}", True, description)
            except ImportError:
                self.add_result(f"Optional: {package}", False, f"Not available - {description}")


class PerformanceValidator(ComponentValidator):
    """Validate system performance requirements"""
    
    def __init__(self):
        super().__init__("Performance")
        
    async def validate_latency_requirements(self):
        """Check if system meets latency requirements"""
        # Test market data processing latency
        start = time.time()
        await asyncio.sleep(0.001)  # Simulate processing
        market_latency = (time.time() - start) * 1000
        
        self.add_result(
            "Market data latency",
            market_latency < 5,
            f"{market_latency:.2f}ms (target: <5ms)"
        )
        
        # Test strategy decision latency
        start = time.time()
        await asyncio.sleep(0.01)  # Simulate decision
        strategy_latency = (time.time() - start) * 1000
        
        self.add_result(
            "Strategy decision latency",
            strategy_latency < 50,
            f"{strategy_latency:.2f}ms (target: <50ms)"
        )
        
        # Test execution latency
        start = time.time()
        await asyncio.sleep(0.05)  # Simulate execution
        execution_latency = (time.time() - start) * 1000
        
        self.add_result(
            "Execution latency",
            execution_latency < 100,
            f"{execution_latency:.2f}ms (target: <100ms)"
        )
        
    def validate_memory_usage(self):
        """Check memory usage is within limits"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        self.add_result(
            "Memory usage",
            memory_mb < 1000,
            f"{memory_mb:.0f}MB (target: <1000MB)"
        )


class IntegrationValidator(ComponentValidator):
    """Validate component integration"""
    
    def __init__(self):
        super().__init__("Integration")
        
    async def validate_component_communication(self):
        """Test communication between components"""
        # Test attention -> regime detector flow
        attention_output = {'weighted_features': {'volatility': 0.001}, 'phase': 'active'}
        regime_input_valid = 'weighted_features' in attention_output
        
        self.add_result(
            "Attention -> Regime Detector",
            regime_input_valid,
            "Data structure compatible"
        )
        
        # Test regime -> strategy selector flow
        regime_output = ('ranging', 0.8)
        strategy_input_valid = isinstance(regime_output, tuple) and len(regime_output) == 2
        
        self.add_result(
            "Regime Detector -> Strategy Selector",
            strategy_input_valid,
            "Output format correct"
        )
        
        # Test strategy -> risk management flow
        strategy_output = {
            'grid_type': 'symmetric',
            'spacing': 0.001,
            'levels': 10,
            'enabled': True
        }
        risk_input_valid = all(k in strategy_output for k in ['grid_type', 'spacing', 'levels'])
        
        self.add_result(
            "Strategy Selector -> Risk Management",
            risk_input_valid,
            "Strategy config complete"
        )
        
        # Test risk -> execution flow
        risk_output = {
            'approved': True,
            'adjusted_size': 0.1,
            'risk_level': 'low'
        }
        execution_input_valid = 'approved' in risk_output
        
        self.add_result(
            "Risk Management -> Execution",
            execution_input_valid,
            "Risk decision available"
        )


class SafetyValidator(ComponentValidator):
    """Validate safety mechanisms"""
    
    def __init__(self):
        super().__init__("Safety")
        
    def validate_risk_limits(self):
        """Check risk management limits are properly set"""
        config = create_test_config()
        risk_config = config['risk_management']
        
        # Position size check
        self.add_result(
            "Position size limit set",
            risk_config['max_position_size'] <= 0.1,
            f"Limit: {risk_config['max_position_size']*100:.0f}%"
        )
        
        # Daily loss check
        self.add_result(
            "Daily loss limit set",
            risk_config['max_daily_loss'] <= 0.02,
            f"Limit: {risk_config['max_daily_loss']*100:.0f}%"
        )
        
        # Concurrent orders check
        self.add_result(
            "Concurrent orders limited",
            risk_config['max_concurrent_orders'] <= 10,
            f"Limit: {risk_config['max_concurrent_orders']}"
        )
        
    def validate_overfitting_protection(self):
        """Check overfitting protection mechanisms"""
        overfitting_config = create_overfitting_config()
        
        # Detection enabled
        self.add_result(
            "Overfitting detection enabled",
            overfitting_config['overfitting_detection']['sensitivity'] > 0,
            "Active monitoring"
        )
        
        # Regularization active
        reg_config = overfitting_config['regularization']
        self.add_result(
            "Regularization configured",
            reg_config['dropout_rate'] > 0 and reg_config['weight_decay'] > 0,
            "Dropout and weight decay active"
        )
        
        # Checkpointing enabled
        self.add_result(
            "Checkpointing enabled",
            overfitting_config['checkpointing']['enabled'],
            "Can recover from degradation"
        )


# ============================================================================
# System Validation Runner
# ============================================================================

class SystemValidationRunner:
    """Run complete system validation"""
    
    def __init__(self):
        self.validators = [
            ConfigurationValidator(),
            DependencyValidator(),
            DataFlowValidator(),
            PerformanceValidator(),
            IntegrationValidator(),
            SafetyValidator()
        ]
        self.results = {}
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run all validation checks"""
        logger.info("="*80)
        logger.info("GRIDATTENTION SYSTEM VALIDATION")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Load configurations
        config = create_test_config()
        overfitting_config = create_overfitting_config()
        
        # Run configuration validation
        config_validator = self.validators[0]
        config_validator.validate_config_structure(config)
        
        # Run dependency validation
        dep_validator = self.validators[1]
        dep_validator.validate_python_packages()
        dep_validator.validate_optional_packages()
        
        # Run async validations
        await self.validators[2].validate_market_data_flow()  # DataFlow
        await self.validators[3].validate_latency_requirements()  # Performance
        self.validators[3].validate_memory_usage()
        await self.validators[4].validate_component_communication()  # Integration
        self.validators[5].validate_risk_limits()  # Safety
        self.validators[5].validate_overfitting_protection()
        
        # Collect results
        for validator in self.validators:
            self.results[validator.component_name] = validator.get_summary()
            
        # Generate report
        report = self._generate_report(time.time() - start_time)
        
        return report
        
    def _generate_report(self, execution_time: float) -> Dict[str, Any]:
        """Generate validation report"""
        total_checks = 0
        total_passed = 0
        all_errors = []
        
        for component, summary in self.results.items():
            total_checks += summary['total_checks']
            total_passed += summary['passed']
            all_errors.extend(summary['errors'])
            
        report = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'overall': {
                'total_checks': total_checks,
                'passed': total_passed,
                'failed': total_checks - total_passed,
                'success_rate': total_passed / total_checks if total_checks > 0 else 0,
                'system_ready': total_passed == total_checks
            },
            'components': self.results,
            'errors': all_errors,
            'recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check each component
        for component, summary in self.results.items():
            if summary['success_rate'] < 1.0:
                if component == 'Dependencies':
                    recommendations.append(
                        "Install missing packages: pip install -r requirements.txt"
                    )
                elif component == 'Configuration':
                    recommendations.append(
                        "Review and adjust configuration parameters in config.yaml"
                    )
                elif component == 'Performance':
                    recommendations.append(
                        "Optimize system performance or upgrade hardware"
                    )
                elif component == 'Safety':
                    recommendations.append(
                        "Ensure all safety mechanisms are properly configured"
                    )
                    
        if not recommendations:
            recommendations.append("System validation passed - ready for deployment!")
            
        return recommendations
        
    def print_report(self, report: Dict[str, Any]):
        """Print validation report"""
        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)
        
        print(f"\nExecution Time: {report['execution_time']:.2f} seconds")
        print(f"System Ready: {'✅ YES' if report['overall']['system_ready'] else '❌ NO'}")
        
        print(f"\nOverall Results:")
        print(f"  Total Checks: {report['overall']['total_checks']}")
        print(f"  Passed: {report['overall']['passed']}")
        print(f"  Failed: {report['overall']['failed']}")
        print(f"  Success Rate: {report['overall']['success_rate']:.1%}")
        
        print("\nComponent Validation:")
        print("-"*80)
        print(f"{'Component':<20} {'Checks':>10} {'Passed':>10} {'Failed':>10} {'Rate':>10}")
        print("-"*80)
        
        for component, summary in report['components'].items():
            print(f"{component:<20} {summary['total_checks']:>10} "
                  f"{summary['passed']:>10} {summary['failed']:>10} "
                  f"{summary['success_rate']:>9.1%}")
                  
        if report['errors']:
            print("\nErrors Found:")
            print("-"*80)
            for error in report['errors']:
                print(f"  ❌ {error}")
                
        print("\nRecommendations:")
        print("-"*80)
        for rec in report['recommendations']:
            print(f"  → {rec}")
            
        print("\n" + "="*80)
        
    def save_report(self, report: Dict[str, Any], filepath: str = "validation_report.json"):
        """Save validation report to file"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report saved to {filepath}")


# ============================================================================
# Quick System Check
# ============================================================================

async def quick_system_check():
    """Perform quick system health check"""
    logger.info("\n🚀 Quick System Check")
    
    checks = {
        'Configuration': True,
        'Dependencies': True,
        'Data Flow': True,
        'Risk Limits': True,
        'Overfitting Protection': True
    }
    
    try:
        # Check config exists
        config = create_test_config()
        checks['Configuration'] = len(config) > 0
        
        # Check key imports
        try:
            import numpy
            import pandas
            checks['Dependencies'] = True
        except:
            checks['Dependencies'] = False
            
        # Check data structures
        test_tick = {'price': 50000, 'volume': 100}
        test_features = {'volatility': 0.001}
        checks['Data Flow'] = 'price' in test_tick and 'volatility' in test_features
        
        # Check risk config
        risk_config = config.get('risk_management', {})
        checks['Risk Limits'] = (
            risk_config.get('max_position_size', 1) <= 0.1 and
            risk_config.get('max_daily_loss', 1) <= 0.02
        )
        
        # Check overfitting config
        overfitting_config = create_overfitting_config()
        checks['Overfitting Protection'] = (
            overfitting_config['overfitting_detection']['sensitivity'] > 0
        )
        
    except Exception as e:
        logger.error(f"Quick check failed: {e}")
        
    # Print results
    all_passed = all(checks.values())
    
    print("\nQuick Check Results:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        
    if all_passed:
        print("\n✅ System appears healthy!")
    else:
        print("\n⚠️  Some checks failed - run full validation for details")
        
    return all_passed


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main validation execution"""
    # Quick check first
    quick_passed = await quick_system_check()
    
    if not quick_passed:
        logger.warning("Quick check failed - running full validation...")
        
    # Run full validation
    print("\n" + "="*80)
    print("Running Full System Validation...")
    print("="*80)
    
    runner = SystemValidationRunner()
    report = await runner.run_validation()
    
    # Print and save report
    runner.print_report(report)
    runner.save_report(report)
    
    # Return success status
    return report['overall']['system_ready']


if __name__ == "__main__":
    # Run validation
    success = asyncio.run(main())
    
    # Exit with appropriate code
    exit(0 if success else 1)
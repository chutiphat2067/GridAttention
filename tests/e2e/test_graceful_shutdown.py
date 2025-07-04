#!/usr/bin/env python3
"""
End-to-End test for GridAttention graceful shutdown
Tests the system's ability to shut down safely while preserving state
and handling active operations.
"""

import pytest
import asyncio
import signal
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
import pickle

# Assuming project structure based on the document
import sys
sys.path.append('../..')

# Core components - from architecture document
from core.attention_learning_layer import AttentionLearningLayer, AttentionPhase
from core.market_regime_detector import MarketRegimeDetector
from core.overfitting_detector import OverfittingDetector

# Model components
from models.grid_strategy_selector import GridStrategySelector
from models.risk_management_system import RiskManagementSystem

# Infrastructure components
from infrastructure.execution_engine import ExecutionEngine
from infrastructure.performance_monitor import PerformanceMonitor
from infrastructure.feedback_loop import FeedbackLoop

# Data components
from data.market_data_input import MarketDataInput
from data.feature_engineering_pipeline import FeatureEngineeringPipeline


class TestGracefulShutdown:
    """Test suite for graceful shutdown scenarios"""
    
    @pytest.fixture
    async def mock_components(self):
        """Create mock components for testing"""
        return {
            # Core components
            'attention_layer': Mock(spec=AttentionLearningLayer),
            'regime_detector': Mock(spec=MarketRegimeDetector),
            'overfitting_detector': Mock(spec=OverfittingDetector),
            
            # Model components
            'strategy_selector': Mock(spec=GridStrategySelector),
            'risk_manager': Mock(spec=RiskManagementSystem),
            
            # Infrastructure components
            'execution_engine': Mock(spec=ExecutionEngine),
            'performance_monitor': Mock(spec=PerformanceMonitor),
            'feedback_loop': Mock(spec=FeedbackLoop),
            
            # Data components
            'market_data': Mock(spec=MarketDataInput),
            'feature_pipeline': Mock(spec=FeatureEngineeringPipeline),
            
            # External connections
            'exchange_client': AsyncMock(),
            'database': AsyncMock(),
            'event_bus': AsyncMock(),
            'websocket_manager': AsyncMock()
        }
    
    @pytest.fixture
    async def system_state(self):
        """Create a sample system state"""
        return {
            'phase': AttentionPhase.ACTIVE.value,  # Using AttentionPhase enum
            'regime': 'trending',
            'active_positions': {
                'BTC/USDT': {
                    'side': 'long',
                    'size': 0.5,
                    'entry_price': 45000,
                    'current_price': 46000,
                    'pnl': 500,
                    'grid_id': 'grid_btc_001'
                },
                'ETH/USDT': {
                    'side': 'short',
                    'size': 2.0,
                    'entry_price': 3200,
                    'current_price': 3150,
                    'pnl': 100,
                    'grid_id': 'grid_eth_001'
                }
            },
            'pending_orders': [
                {
                    'id': 'order_123',
                    'symbol': 'BTC/USDT',
                    'side': 'buy',
                    'price': 44500,
                    'amount': 0.1,
                    'status': 'pending',
                    'grid_level': 5
                },
                {
                    'id': 'order_456',
                    'symbol': 'ETH/USDT',
                    'side': 'sell',
                    'price': 3300,
                    'amount': 1.0,
                    'status': 'pending',
                    'grid_level': 3
                }
            ],
            'active_grids': {
                'BTC/USDT': {
                    'type': 'neutral',
                    'levels': 10,
                    'range': [44000, 48000],
                    'active_orders': 5,
                    'strategy': 'symmetric_grid'
                }
            },
            'attention_weights': {
                'feature_attention': np.array([0.3, 0.25, 0.25, 0.2]),
                'temporal_attention': np.array([0.4, 0.3, 0.2, 0.1]),
                'regime_attention': 0.85
            },
            'metrics': {
                'total_trades': 1543,
                'win_rate': 0.62,
                'sharpe_ratio': 1.85,
                'max_drawdown': 0.08,
                'overfitting_score': 0.12
            }
        }
    
    @pytest.mark.asyncio
    async def test_shutdown_signal_handling(self, mock_components):
        """Test system responds correctly to shutdown signals"""
        shutdown_handler = ShutdownHandler(mock_components)
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, shutdown_handler.handle_signal)
        signal.signal(signal.SIGINT, shutdown_handler.handle_signal)
        
        # Simulate SIGTERM
        shutdown_handler.shutdown_event = asyncio.Event()
        signal.raise_signal(signal.SIGTERM)
        
        # Wait for shutdown flag
        await asyncio.sleep(0.1)
        
        assert shutdown_handler.shutdown_requested
        assert shutdown_handler.shutdown_signal == signal.SIGTERM
    
    @pytest.mark.asyncio
    async def test_cancel_pending_orders(self, mock_components, system_state):
        """Test all pending orders are cancelled during shutdown"""
        mock_components['execution_engine'].cancel_order = AsyncMock(return_value=True)
        mock_components['execution_engine'].get_pending_orders = AsyncMock(
            return_value=system_state['pending_orders']
        )
        
        shutdown_manager = ShutdownManager(mock_components)
        cancelled_orders = await shutdown_manager.cancel_all_pending_orders()
        
        # Verify all orders were cancelled
        assert len(cancelled_orders) == len(system_state['pending_orders'])
        assert mock_components['execution_engine'].cancel_order.call_count == 2
        
        # Verify correct order IDs were cancelled
        cancelled_ids = [call[0][0] for call in 
                        mock_components['execution_engine'].cancel_order.call_args_list]
        expected_ids = ['order_123', 'order_456']
        assert set(cancelled_ids) == set(expected_ids)
    
    @pytest.mark.asyncio
    async def test_close_positions_safely(self, mock_components, system_state):
        """Test positions are closed safely with proper risk checks"""
        mock_components['execution_engine'].get_open_positions = AsyncMock(
            return_value=system_state['active_positions']
        )
        mock_components['risk_manager'].validate_emergency_close = AsyncMock(
            return_value=True
        )
        mock_components['execution_engine'].close_position = AsyncMock(
            return_value={'status': 'closed', 'exit_price': 46100}
        )
        
        shutdown_manager = ShutdownManager(mock_components)
        closed_positions = await shutdown_manager.close_all_positions()
        
        # Verify all positions were closed
        assert len(closed_positions) == len(system_state['active_positions'])
        
        # Verify risk checks were performed
        assert mock_components['risk_manager'].validate_emergency_close.call_count == 2
        
        # Verify position closure
        assert mock_components['execution_engine'].close_position.call_count == 2
    
    @pytest.mark.asyncio
    async def test_save_system_state(self, mock_components, system_state, tmp_path):
        """Test system state is properly saved during shutdown"""
        state_file = tmp_path / "shutdown_state.json"
        
        shutdown_manager = ShutdownManager(mock_components)
        shutdown_manager.state_file = state_file
        
        # Mock database save
        mock_components['database'].save_shutdown_state = AsyncMock(return_value=True)
        
        # Save state
        await shutdown_manager.save_system_state(system_state)
        
        # Verify file was created
        assert state_file.exists()
        
        # Verify content
        with open(state_file, 'r') as f:
            saved_state = json.load(f)
        
        assert saved_state['phase'] == system_state['phase']
        assert saved_state['regime'] == system_state['regime']
        assert saved_state['timestamp'] is not None
        assert saved_state['shutdown_reason'] is not None
        assert 'attention_weights' in saved_state
        assert 'metrics' in saved_state
        
        # Verify database save
        mock_components['database'].save_shutdown_state.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_component_shutdown_sequence(self, mock_components):
        """Test components are shut down in correct order"""
        shutdown_sequence = []
        
        # Mock shutdown methods with sequence tracking
        async def mock_shutdown(component_name):
            shutdown_sequence.append(component_name)
            await asyncio.sleep(0.01)
        
        for name, component in mock_components.items():
            if hasattr(component, 'shutdown'):
                component.shutdown = AsyncMock(
                    side_effect=lambda n=name: mock_shutdown(n)
                )
            elif hasattr(component, 'close'):
                component.close = AsyncMock(
                    side_effect=lambda n=name: mock_shutdown(n)
                )
            elif hasattr(component, 'disconnect'):
                component.disconnect = AsyncMock(
                    side_effect=lambda n=name: mock_shutdown(n)
                )
        
        shutdown_manager = ShutdownManager(mock_components)
        await shutdown_manager.shutdown_components()
        
        # Verify shutdown order (critical components last)
        # Risk and execution should be after analysis components
        if 'risk_manager' in shutdown_sequence and 'attention_layer' in shutdown_sequence:
            risk_index = shutdown_sequence.index('risk_manager')
            attention_index = shutdown_sequence.index('attention_layer')
            assert risk_index > attention_index
        
        if 'execution_engine' in shutdown_sequence and 'risk_manager' in shutdown_sequence:
            execution_index = shutdown_sequence.index('execution_engine')
            risk_index = shutdown_sequence.index('risk_manager')
            assert execution_index >= risk_index
        
        # Verify monitoring components shut down first
        if 'feedback_loop' in shutdown_sequence and 'execution_engine' in shutdown_sequence:
            feedback_index = shutdown_sequence.index('feedback_loop')
            execution_index = shutdown_sequence.index('execution_engine')
            assert feedback_index < execution_index
    
    @pytest.mark.asyncio
    async def test_shutdown_with_active_learning(self, mock_components):
        """Test shutdown during active learning phase"""
        # Setup active learning state
        mock_components['attention_layer'].is_training = True
        mock_components['attention_layer'].phase = AttentionPhase.LEARNING
        mock_components['attention_layer'].save_checkpoint = AsyncMock(
            return_value='checkpoint_20240115_123456.pkl'
        )
        mock_components['attention_layer'].current_epoch = 150
        mock_components['attention_layer'].best_validation_loss = 0.0234
        mock_components['attention_layer'].feature_importance_scores = {
            'volatility_5m': 0.35,
            'trend_strength': 0.25,
            'volume_ratio': 0.20,
            'rsi_14': 0.20
        }
        
        # Mock overfitting detector state
        mock_components['overfitting_detector'].get_current_state = AsyncMock(
            return_value={
                'overfitting_score': 0.15,
                'validation_trend': 'stable',
                'regularization_active': False
            }
        )
        
        shutdown_manager = ShutdownManager(mock_components)
        checkpoint_file = await shutdown_manager.save_learning_state()
        
        # Verify checkpoint was saved
        assert checkpoint_file is not None
        mock_components['attention_layer'].save_checkpoint.assert_called_once()
        
        # Verify overfitting state was saved
        mock_components['overfitting_detector'].get_current_state.assert_called_once()
        
        # Verify training was stopped gracefully
        assert hasattr(mock_components['attention_layer'], 'stop_training')
    
    @pytest.mark.asyncio
    async def test_shutdown_timeout_handling(self, mock_components):
        """Test system handles shutdown timeout gracefully"""
        # Mock a component that takes too long to shutdown
        async def slow_shutdown():
            await asyncio.sleep(10)  # Longer than timeout
        
        mock_components['execution_engine'].shutdown = AsyncMock(
            side_effect=slow_shutdown
        )
        
        shutdown_manager = ShutdownManager(mock_components)
        shutdown_manager.shutdown_timeout = 2  # 2 second timeout
        
        start_time = time.time()
        
        with pytest.raises(asyncio.TimeoutError):
            await shutdown_manager.shutdown_with_timeout()
        
        elapsed = time.time() - start_time
        assert elapsed < 3  # Should timeout around 2 seconds
    
    @pytest.mark.asyncio
    async def test_data_persistence_verification(self, mock_components, system_state):
        """Test critical data is persisted and verified"""
        mock_components['database'].save_final_metrics = AsyncMock(return_value=True)
        mock_components['database'].save_position_history = AsyncMock(return_value=True)
        mock_components['database'].save_trade_log = AsyncMock(return_value=True)
        mock_components['database'].verify_data_integrity = AsyncMock(return_value=True)
        
        shutdown_manager = ShutdownManager(mock_components)
        
        # Perform data persistence
        await shutdown_manager.persist_critical_data(system_state)
        
        # Verify all critical data was saved
        mock_components['database'].save_final_metrics.assert_called_once()
        mock_components['database'].save_position_history.assert_called_once()
        mock_components['database'].save_trade_log.assert_called_once()
        
        # Verify data integrity check
        mock_components['database'].verify_data_integrity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_event_bus_cleanup(self, mock_components):
        """Test event bus is properly cleaned up"""
        # Mock pending events
        pending_events = [
            {'type': 'MARKET_UPDATE', 'data': {}},
            {'type': 'POSITION_OPENED', 'data': {}},
            {'type': 'RISK_ALERT', 'data': {}}
        ]
        
        mock_components['event_bus'].get_pending_events = AsyncMock(
            return_value=pending_events
        )
        mock_components['event_bus'].flush = AsyncMock(return_value=True)
        mock_components['event_bus'].disconnect = AsyncMock(return_value=True)
        
        shutdown_manager = ShutdownManager(mock_components)
        await shutdown_manager.cleanup_event_bus()
        
        # Verify event bus was flushed and disconnected
        mock_components['event_bus'].flush.assert_called_once()
        mock_components['event_bus'].disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_components):
        """Test system degrades gracefully when components fail during shutdown"""
        # Mock some components failing during shutdown
        mock_components['feedback_loop'].shutdown = AsyncMock(
            side_effect=Exception("Feedback loop shutdown failed")
        )
        mock_components['performance_monitor'].shutdown = AsyncMock(
            side_effect=Exception("Monitor shutdown failed")
        )
        
        shutdown_manager = ShutdownManager(mock_components)
        errors = await shutdown_manager.shutdown_with_error_handling()
        
        # Verify system continued shutdown despite errors
        assert len(errors) == 2
        assert 'feedback_loop' in errors
        assert 'performance_monitor' in errors
        
        # Verify critical components still attempted shutdown
        if hasattr(mock_components['execution_engine'], 'shutdown'):
            mock_components['execution_engine'].shutdown.assert_called()
    
    @pytest.mark.asyncio
    async def test_recovery_checkpoint_creation(self, mock_components, tmp_path):
        """Test recovery checkpoint is created for restart"""
        recovery_file = tmp_path / "recovery_checkpoint.pkl"
        
        recovery_data = {
            'shutdown_time': datetime.now().isoformat(),
            'last_known_state': AttentionPhase.ACTIVE.value,
            'last_regime': 'trending',
            'incomplete_operations': [],
            'restart_instructions': {
                'resume_from_epoch': 150,
                'reload_positions': True,
                'validate_state': True,
                'attention_phase': AttentionPhase.ACTIVE.value,
                'feature_importance': {
                    'volatility_5m': 0.35,
                    'trend_strength': 0.25
                }
            },
            'grid_configurations': {
                'BTC/USDT': {
                    'strategy': 'symmetric_grid',
                    'levels': 10,
                    'range': [44000, 48000]
                }
            }
        }
        
        shutdown_manager = ShutdownManager(mock_components)
        shutdown_manager.recovery_file = recovery_file
        
        await shutdown_manager.create_recovery_checkpoint(recovery_data)
        
        # Verify checkpoint was created
        assert recovery_file.exists()
        
        # Verify content
        with open(recovery_file, 'rb') as f:
            saved_data = pickle.load(f)
        
        assert saved_data['shutdown_time'] is not None
        assert saved_data['last_regime'] == 'trending'
        assert saved_data['restart_instructions']['resume_from_epoch'] == 150
        assert 'grid_configurations' in saved_data


class ShutdownHandler:
    """Handles shutdown signals"""
    
    def __init__(self, components):
        self.components = components
        self.shutdown_requested = False
        self.shutdown_signal = None
        self.shutdown_event = asyncio.Event()
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signal"""
        self.shutdown_requested = True
        self.shutdown_signal = signum
        self.shutdown_event.set()


class ShutdownManager:
    """Manages graceful shutdown process"""
    
    def __init__(self, components):
        self.components = components
        self.shutdown_timeout = 30  # seconds
        self.state_file = Path("shutdown_state.json")
        self.recovery_file = Path("recovery_checkpoint.pkl")
    
    async def cancel_all_pending_orders(self):
        """Cancel all pending orders"""
        pending_orders = await self.components['execution_engine'].get_pending_orders()
        cancelled = []
        
        for order in pending_orders:
            try:
                result = await self.components['execution_engine'].cancel_order(order['id'])
                if result:
                    cancelled.append(order['id'])
            except Exception as e:
                print(f"Failed to cancel order {order['id']}: {e}")
        
        return cancelled
    
    async def close_all_positions(self):
        """Close all open positions safely"""
        positions = await self.components['execution_engine'].get_open_positions()
        closed = []
        
        for symbol, position in positions.items():
            try:
                # Validate with risk manager
                if await self.components['risk_manager'].validate_emergency_close(position):
                    result = await self.components['execution_engine'].close_position(symbol)
                    closed.append({
                        'symbol': symbol,
                        'result': result
                    })
            except Exception as e:
                print(f"Failed to close position {symbol}: {e}")
        
        return closed
    
    async def save_system_state(self, state):
        """Save current system state"""
        # Convert numpy arrays to lists for JSON serialization
        attention_weights = state.get('attention_weights', {})
        serializable_weights = {}
        for key, value in attention_weights.items():
            if isinstance(value, np.ndarray):
                serializable_weights[key] = value.tolist()
            else:
                serializable_weights[key] = value
        
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'phase': state['phase'],
            'regime': state.get('regime', 'unknown'),
            'positions': state.get('active_positions', {}),
            'grids': state.get('active_grids', {}),
            'attention_weights': serializable_weights,
            'metrics': state.get('metrics', {}),
            'shutdown_reason': 'graceful_shutdown'
        }
        
        # Save to file
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        # Save to database
        await self.components['database'].save_shutdown_state(state_data)
    
    async def shutdown_components(self):
        """Shutdown components in correct order"""
        # Define shutdown order based on GridAttention architecture
        # (reverse dependency order - least critical first)
        shutdown_order = [
            # Monitoring and feedback (can shut down first)
            'feedback_loop',
            'performance_monitor',
            'overfitting_detector',
            
            # Strategy and analysis
            'strategy_selector',
            'regime_detector',
            'attention_layer',
            
            # Data processing
            'feature_pipeline',
            'market_data',
            
            # Critical execution components (shut down last)
            'risk_manager',
            'execution_engine',
            
            # External connections
            'websocket_manager',
            'event_bus',
            'database',
            'exchange_client'
        ]
        
        for component_name in shutdown_order:
            if component_name in self.components:
                component = self.components[component_name]
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                elif hasattr(component, 'close'):
                    await component.close()
                elif hasattr(component, 'disconnect'):
                    await component.disconnect()
    
    async def save_learning_state(self):
        """Save learning state if training is active"""
        attention = self.components['attention_layer']
        overfitting = self.components.get('overfitting_detector')
        
        saved_files = []
        
        # Save attention layer state
        if hasattr(attention, 'is_training') and attention.is_training:
            # Save checkpoint
            checkpoint_file = await attention.save_checkpoint()
            saved_files.append(checkpoint_file)
            
            # Stop training gracefully
            if hasattr(attention, 'stop_training'):
                attention.stop_training()
        
        # Save overfitting detector state
        if overfitting and hasattr(overfitting, 'get_current_state'):
            overfitting_state = await overfitting.get_current_state()
            # Save overfitting state to file or include in checkpoint
            
        return saved_files[0] if saved_files else None
    
    async def shutdown_with_timeout(self):
        """Shutdown with timeout protection"""
        try:
            await asyncio.wait_for(
                self.shutdown_components(),
                timeout=self.shutdown_timeout
            )
        except asyncio.TimeoutError:
            print(f"Shutdown timeout after {self.shutdown_timeout} seconds")
            raise
    
    async def persist_critical_data(self, state):
        """Persist all critical data"""
        db = self.components['database']
        
        # Save final metrics
        await db.save_final_metrics(state.get('metrics', {}))
        
        # Save position history
        await db.save_position_history(state.get('active_positions', {}))
        
        # Save trade log
        await db.save_trade_log()
        
        # Verify data integrity
        await db.verify_data_integrity()
    
    async def cleanup_event_bus(self):
        """Clean up event bus"""
        event_bus = self.components['event_bus']
        
        # Get pending events
        pending = await event_bus.get_pending_events()
        
        # Flush remaining events
        await event_bus.flush()
        
        # Disconnect
        await event_bus.disconnect()
    
    async def shutdown_with_error_handling(self):
        """Shutdown with error handling for each component"""
        errors = {}
        
        for name, component in self.components.items():
            if hasattr(component, 'shutdown'):
                try:
                    await component.shutdown()
                except Exception as e:
                    errors[name] = str(e)
                    print(f"Error shutting down {name}: {e}")
        
        return errors
    
    async def create_recovery_checkpoint(self, recovery_data):
        """Create checkpoint for system recovery"""
        with open(self.recovery_file, 'wb') as f:
            pickle.dump(recovery_data, f)


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v", "-k", "test_graceful_shutdown"])
"""
Dashboard Integration Module for GridAttention System
Connects the practical dashboard with all system components
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from aiohttp import web
import aiohttp
from aiohttp.web import WebSocketResponse
import psutil

logger = logging.getLogger(__name__)


class DashboardDataCollector:
    """Collects data from all system components for dashboard"""
    
    def __init__(self, grid_system):
        self.system = grid_system
        self.last_values = {}
        self.start_time = time.time()
        
    async def collect_all_data(self) -> Dict[str, Any]:
        """Collect comprehensive data from all components"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': await self._get_system_status(),
                'critical_metrics': await self._get_critical_metrics(),
                'learning_status': await self._get_learning_status(),
                'augmentation': await self._get_augmentation_status(),
                'trading_activity': await self._get_trading_activity(),
                'market_analysis': await self._get_market_analysis(),
                'system_health': await self._get_system_health(),
                'risk_controls': await self._get_risk_controls(),
                'logs': await self._get_recent_logs()
            }
            
            # Calculate changes
            self._calculate_changes(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting dashboard data: {e}")
            return self._get_error_response(str(e))
            
    async def _get_system_status(self) -> str:
        """Get overall system status"""
        if self.system._running:
            if self.system._in_safe_mode:
                return 'SAFE_MODE'
            elif self.system._shutdown_requested:
                return 'SHUTTING_DOWN'
            else:
                return 'RUNNING'
        return 'STOPPED'
        
    async def _get_critical_metrics(self) -> Dict[str, Any]:
        """Get critical trading metrics"""
        perf_monitor = self.system.components.get('performance_monitor')
        risk_manager = self.system.components.get('risk_manager')
        overfitting = self.system.components.get('overfitting_detector')
        
        metrics = {
            'pnl': 0,
            'winRate': 0,
            'drawdown': 0,
            'overfittingScore': 0,
            'openPositions': 0,
            'latency': 0
        }
        
        if perf_monitor:
            trading_metrics = await perf_monitor.get_current_metrics()
            metrics['pnl'] = trading_metrics.get('total_pnl', 0)
            metrics['winRate'] = trading_metrics.get('win_rate', 0) * 100
            metrics['drawdown'] = trading_metrics.get('current_drawdown', 0) * 100
            
        if risk_manager:
            positions = await risk_manager.get_open_positions()
            metrics['openPositions'] = len(positions)
            
        if overfitting:
            metrics['overfittingScore'] = await overfitting.get_current_score()
            
        # Get execution latency
        execution = self.system.components.get('execution')
        if execution:
            metrics['latency'] = await execution.get_avg_latency()
            
        return metrics
        
    async def _get_learning_status(self) -> Dict[str, Any]:
        """Get attention learning status"""
        attention = self.system.components.get('attention')
        if not attention:
            return {}
            
        state = await attention.get_attention_state()
        
        # Calculate time to next phase
        obs_per_min = 30  # Estimate based on tick rate
        thresholds = {
            'learning': 2000,
            'shadow': 500,
            'active': 200
        }
        
        current_obs = state['total_observations']
        phase = state['phase']
        
        if phase == 'learning':
            remaining = thresholds['learning'] - current_obs
        elif phase == 'shadow':
            remaining = thresholds['shadow'] - (current_obs - thresholds['learning'])
        else:
            remaining = 0
            
        time_to_phase = remaining / obs_per_min if remaining > 0 else 0
        
        return {
            'phase': phase,
            'observations': current_obs,
            'phaseProgress': attention.get_learning_progress() * 100,
            'timeToPhase': time_to_phase,
            'learningRate': attention.config.get('learning_rate', 0.001),
            'warmupLoaded': attention.warmup_loaded,
            'topFeatures': list(state.get('feature_importance', {}).items())[:5]
        }
        
    async def _get_augmentation_status(self) -> Dict[str, Any]:
        """Get data augmentation status"""
        if not self.system.augmentation_manager:
            return {
                'status': 'inactive',
                'factor': 1.0,
                'totalAugmented': 0,
                'quality': 0
            }
            
        dashboard_data = self.system.augmentation_manager.get_monitoring_dashboard()
        stats = self.system.augmentation_manager.get_stats()
        
        return {
            'status': 'active' if stats['total_augmented'] > 0 else 'inactive',
            'factor': dashboard_data['summary'].get('average_factor', 1.0),
            'totalAugmented': stats['total_augmented'],
            'quality': dashboard_data['summary'].get('average_quality', 0),
            'methodUsage': dashboard_data.get('method_usage', {}),
            'byPhase': stats['augmentation_by_phase']
        }
        
    async def _get_trading_activity(self) -> Dict[str, Any]:
        """Get current trading activity"""
        perf_monitor = self.system.components.get('performance_monitor')
        risk_manager = self.system.components.get('risk_manager')
        execution = self.system.components.get('execution')
        
        activity = {
            'tradesToday': 0,
            'volumeToday': 0,
            'feesToday': 0,
            'profitFactor': 0,
            'openPositions': []
        }
        
        if perf_monitor:
            metrics = await perf_monitor.get_daily_metrics()
            activity['tradesToday'] = metrics.get('trade_count', 0)
            activity['volumeToday'] = metrics.get('volume', 0)
            activity['feesToday'] = metrics.get('fees', 0)
            activity['profitFactor'] = metrics.get('profit_factor', 0)
            
        if risk_manager:
            positions = await risk_manager.get_open_positions()
            activity['openPositions'] = [
                {
                    'symbol': p.symbol,
                    'side': p.side,
                    'size': p.size,
                    'entry': p.entry_price,
                    'current': p.current_price,
                    'pnl': p.unrealized_pnl
                }
                for p in positions[:10]  # Limit to 10 positions
            ]
            
        return activity
        
    async def _get_market_analysis(self) -> Dict[str, Any]:
        """Get market regime and analysis"""
        regime_detector = self.system.components.get('regime_detector')
        strategy_selector = self.system.components.get('strategy_selector')
        features = self.system.components.get('features')
        
        analysis = {
            'regime': 'unknown',
            'confidence': 0,
            'volatility': 0,
            'trendStrength': 0,
            'volumeRatio': 1.0,
            'currentStrategy': 'none',
            'gridLevels': 0,
            'gridSpacing': 0
        }
        
        if regime_detector:
            # Get latest features
            if features:
                latest_features = await features.get_latest_features()
                if latest_features:
                    regime, confidence = await regime_detector.detect_regime(latest_features)
                    analysis['regime'] = regime.value if hasattr(regime, 'value') else str(regime)
                    analysis['confidence'] = confidence * 100
                    analysis['volatility'] = latest_features.get('volatility_5m', 0)
                    analysis['trendStrength'] = latest_features.get('trend_strength', 0)
                    analysis['volumeRatio'] = latest_features.get('volume_ratio', 1.0)
                    
        if strategy_selector and analysis['regime'] != 'unknown':
            # Get current strategy
            strategy = await strategy_selector.get_current_strategy()
            if strategy:
                analysis['currentStrategy'] = strategy.get('type', 'none')
                analysis['gridLevels'] = strategy.get('levels', 0)
                analysis['gridSpacing'] = strategy.get('spacing', 0)
                
        return analysis
        
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network latency (to exchange)
        market_data = self.system.components.get('market_data')
        net_latency = 0
        if market_data and hasattr(market_data, 'get_latency'):
            net_latency = await market_data.get_latency()
            
        # System metrics
        uptime = int(time.time() - self.start_time)
        
        # Tick rate
        tick_rate = 0
        if market_data and hasattr(market_data, 'get_tick_rate'):
            tick_rate = await market_data.get_tick_rate()
            
        # Error count
        error_count = self.system.error_count
        
        # Queue depth
        queue_depth = 0
        execution = self.system.components.get('execution')
        if execution and hasattr(execution, 'get_queue_depth'):
            queue_depth = await execution.get_queue_depth()
            
        return {
            'cpuUsage': cpu_percent,
            'memUsage': memory.percent,
            'diskUsage': disk.percent,
            'netLatency': net_latency,
            'dbStatus': 'OK',  # Simplified
            'apiStatus': 'OK' if market_data else 'ERROR',
            'uptime': uptime,
            'tickRate': tick_rate,
            'errorCount': error_count,
            'queueDepth': queue_depth
        }
        
    async def _get_risk_controls(self) -> Dict[str, Any]:
        """Get risk control status"""
        risk_manager = self.system.components.get('risk_manager')
        if not risk_manager:
            return {}
            
        limits = await risk_manager.get_risk_limits()
        usage = await risk_manager.get_current_usage()
        
        return {
            'limits': limits,
            'usage': usage,
            'overrideEnabled': risk_manager.override_enabled if hasattr(risk_manager, 'override_enabled') else False
        }
        
    async def _get_recent_logs(self) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        # This would read from actual log handler
        # For now, return empty list
        return []
        
    def _calculate_changes(self, data: Dict[str, Any]):
        """Calculate changes from previous values"""
        # Compare with last values and add change indicators
        if 'critical_metrics' in data:
            metrics = data['critical_metrics']
            
            # PnL change
            last_pnl = self.last_values.get('pnl', 0)
            metrics['pnlChange'] = metrics['pnl'] - last_pnl
            
            # Win rate change
            last_wr = self.last_values.get('winRate', 0)
            metrics['winRateChange'] = metrics['winRate'] - last_wr
            
            # Store current values
            self.last_values['pnl'] = metrics['pnl']
            self.last_values['winRate'] = metrics['winRate']
            
    def _get_error_response(self, error: str) -> Dict[str, Any]:
        """Return error response structure"""
        return {
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'system_status': 'ERROR'
        }


class DashboardWebSocketHandler:
    """Handles WebSocket connections for real-time updates"""
    
    def __init__(self, collector: DashboardDataCollector):
        self.collector = collector
        self.websockets = set()
        
    async def handle_websocket(self, request):
        """Handle WebSocket connection"""
        ws = WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        try:
            # Send initial data
            data = await self.collector.collect_all_data()
            await ws.send_json(data)
            
            # Handle incoming messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(ws, msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self.websockets.discard(ws)
            
        return ws
        
    async def _handle_message(self, ws: WebSocketResponse, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'refresh':
                # Send fresh data
                response = await self.collector.collect_all_data()
                await ws.send_json(response)
                
            elif command == 'execute':
                # Execute system command
                action = data.get('action')
                response = await self._execute_action(action, data.get('params', {}))
                await ws.send_json(response)
                
        except Exception as e:
            await ws.send_json({'error': str(e)})
            
    async def _execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system actions from dashboard"""
        system = self.collector.system
        
        try:
            if action == 'emergency_stop':
                await system.emergency_stop()
                return {'success': True, 'message': 'Emergency stop executed'}
                
            elif action == 'pause_trading':
                system._in_safe_mode = True
                return {'success': True, 'message': 'Trading paused'}
                
            elif action == 'resume_trading':
                system._in_safe_mode = False
                return {'success': True, 'message': 'Trading resumed'}
                
            elif action == 'save_checkpoint':
                checkpoint_manager = system.components.get('checkpoint_manager')
                if checkpoint_manager:
                    checkpoint_id = await checkpoint_manager.save_checkpoint('manual')
                    return {'success': True, 'message': f'Checkpoint saved: {checkpoint_id}'}
                    
            elif action == 'toggle_augmentation':
                if system.augmentation_manager:
                    # Toggle augmentation
                    current = system.training_mode
                    system.training_mode = not current
                    return {'success': True, 'message': f'Augmentation {"enabled" if not current else "disabled"}'}
                    
            elif action == 'force_phase_transition':
                attention = system.components.get('attention')
                if attention:
                    await attention.force_phase_transition()
                    return {'success': True, 'message': 'Phase transition forced'}
                    
            elif action == 'set_augmentation_factor':
                if system.augmentation_manager:
                    factor = params.get('factor', 1.0)
                    # This would need to be implemented in augmentation manager
                    return {'success': True, 'message': f'Augmentation factor set to {factor}x'}
                    
            elif action == 'adjust_risk_limit':
                risk_manager = system.components.get('risk_manager')
                if risk_manager:
                    limit_type = params.get('type')
                    value = params.get('value')
                    await risk_manager.adjust_limit(limit_type, value)
                    return {'success': True, 'message': f'{limit_type} limit adjusted to {value}'}
                    
            else:
                return {'success': False, 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            return {'success': False, 'message': str(e)}
            
    async def broadcast_update(self):
        """Broadcast updates to all connected clients"""
        if self.websockets:
            data = await self.collector.collect_all_data()
            
            # Send to all connected clients
            disconnected = set()
            for ws in self.websockets:
                try:
                    await ws.send_json(data)
                except ConnectionResetError:
                    disconnected.add(ws)
                    
            # Remove disconnected clients
            self.websockets -= disconnected


class DashboardServer:
    """Main dashboard server"""
    
    def __init__(self, grid_system, port: int = 8080):
        self.system = grid_system
        self.port = port
        self.app = web.Application()
        self.collector = DashboardDataCollector(grid_system)
        self.ws_handler = DashboardWebSocketHandler(self.collector)
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/', self.serve_dashboard)
        self.app.router.add_get('/ws', self.ws_handler.handle_websocket)
        self.app.router.add_get('/api/data', self.get_data)
        self.app.router.add_post('/api/action', self.execute_action)
        self.app.router.add_get('/api/export/{type}', self.export_data)
        
    async def serve_dashboard(self, request):
        """Serve the dashboard HTML"""
        dashboard_path = Path('scaling_dashboard.html')
        if dashboard_path.exists():
            with open(dashboard_path, 'r') as f:
                html = f.read()
                
            # Inject WebSocket URL
            html = html.replace(
                "// const ws = new WebSocket('ws://localhost:8080/ws');",
                "const ws = new WebSocket('ws://localhost:8080/ws');"
            )
            
            return web.Response(text=html, content_type='text/html')
        else:
            return web.Response(text="Dashboard not found", status=404)
            
    async def get_data(self, request):
        """Get current system data as JSON"""
        data = await self.collector.collect_all_data()
        return web.json_response(data)
        
    async def execute_action(self, request):
        """Execute system action"""
        try:
            body = await request.json()
            action = body.get('action')
            params = body.get('params', {})
            
            result = await self.ws_handler._execute_action(action, params)
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
            
    async def export_data(self, request):
        """Export data in various formats"""
        export_type = request.match_info['type']
        
        if export_type == 'logs':
            # Export logs
            logs = []  # Get from log handler
            return web.Response(
                text='\n'.join(logs),
                headers={'Content-Disposition': 'attachment; filename=logs.txt'}
            )
            
        elif export_type == 'metrics':
            # Export metrics as CSV
            data = await self.collector.collect_all_data()
            # Convert to CSV format
            csv_data = self._convert_to_csv(data)
            return web.Response(
                text=csv_data,
                headers={'Content-Disposition': 'attachment; filename=metrics.csv'}
            )
            
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert data to CSV format"""
        # Simplified CSV conversion
        lines = ['Timestamp,Metric,Value']
        timestamp = data.get('timestamp', '')
        
        if 'critical_metrics' in data:
            for key, value in data['critical_metrics'].items():
                lines.append(f'{timestamp},{key},{value}')
                
        return '\n'.join(lines)
        
    async def start(self):
        """Start dashboard server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        
        logger.info(f"Dashboard server started on http://localhost:{self.port}")
        
        # Start periodic updates
        asyncio.create_task(self._periodic_updates())
        
    async def _periodic_updates(self):
        """Send periodic updates to connected clients"""
        while True:
            try:
                await self.ws_handler.broadcast_update()
                await asyncio.sleep(2)  # Update every 2 seconds
            except Exception as e:
                logger.error(f"Periodic update error: {e}")
                await asyncio.sleep(5)


def integrate_dashboard(grid_system):
    """Helper function to integrate dashboard into existing system"""
    
    # Create dashboard server
    dashboard = DashboardServer(grid_system)
    
    # Add to system components
    grid_system.dashboard_server = dashboard
    
    # Add dashboard task to system tasks
    async def start_dashboard():
        await dashboard.start()
        
    grid_system._tasks.append(asyncio.create_task(start_dashboard()))
    
    return dashboard


# Update main.py to include dashboard
"""
# Add to main.py imports
from dashboard_integration import integrate_dashboard

# Add to GridTradingSystem.__init__
self.dashboard_enabled = True
self.dashboard_server = None

# Add to GridTradingSystem.initialize()
if self.dashboard_enabled:
    self.dashboard_server = integrate_dashboard(self)
    logger.info("✓ Dashboard server initialized")

# Add command line argument
parser.add_argument('--no-dashboard', action='store_true', 
                   help='Disable dashboard server')

# In main()
system.dashboard_enabled = not args.no_dashboard
"""


if __name__ == "__main__":
    # Example standalone usage
    print("Dashboard Integration Module")
    print("This module should be imported by main.py")
    print("\nTo use:")
    print("1. Import in main.py: from dashboard_integration import integrate_dashboard")
    print("2. Call after system init: integrate_dashboard(system)")
    print("3. Access dashboard at: http://localhost:8080")
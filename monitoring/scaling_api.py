"""
scaling_api.py
API endpoints for scaling monitor dashboard

Author: Grid Trading System
Date: 2024
"""

from aiohttp import web
import json
import logging

logger = logging.getLogger(__name__)

class ScalingAPI:
    """API endpoints for scaling monitor"""
    
    def __init__(self, scaling_monitor):
        self.scaling_monitor = scaling_monitor
        self.app = web.Application()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get('/api/scaling-report', self.get_scaling_report)
        self.app.router.add_get('/api/metrics-history', self.get_metrics_history)
        self.app.router.add_post('/api/trigger-analysis', self.trigger_analysis)
        self.app.router.add_get('/api/bottlenecks', self.get_bottlenecks)
        self.app.router.add_get('/augmentation/dashboard', self.augmentation_dashboard)
        self.app.router.add_static('/', path='./static', name='static')
        
    async def get_scaling_report(self, request):
        """Get current scaling report"""
        try:
            report = await self.scaling_monitor.get_scaling_report()
            return web.json_response(report)
        except Exception as e:
            logger.error(f"Error getting scaling report: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def get_metrics_history(self, request):
        """Get historical metrics"""
        try:
            metric_name = request.query.get('metric', 'cpu_usage')
            limit = int(request.query.get('limit', 1000))
            
            history = list(self.scaling_monitor.metric_history.get(metric_name, []))[-limit:]
            
            return web.json_response({
                'metric': metric_name,
                'data': [{'timestamp': h[0], 'value': h[1]} for h in history]
            })
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def trigger_analysis(self, request):
        """Manually trigger scaling analysis"""
        try:
            analysis = await self.scaling_monitor.analyze_scaling_needs()
            return web.json_response(analysis)
        except Exception as e:
            logger.error(f"Error triggering analysis: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def get_bottlenecks(self, request):
        """Get current bottlenecks"""
        try:
            metrics = await self.scaling_monitor.collect_scaling_metrics()
            bottlenecks = self.scaling_monitor.bottleneck_detector.detect(metrics)
            return web.json_response({'bottlenecks': bottlenecks})
        except Exception as e:
            logger.error(f"Error getting bottlenecks: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def augmentation_dashboard(self, request):
        """Get augmentation monitoring dashboard"""
        try:
            # Get reference to grid system from app context
            system = request.app.get('grid_system')
            
            if not system or not system.augmentation_manager:
                return web.json_response({
                    'error': 'Augmentation manager not initialized'
                }, status=400)
                
            # Get dashboard data
            dashboard_data = system.augmentation_manager.get_monitoring_dashboard()
            
            # Return JSON data
            return web.json_response(dashboard_data)
            
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return web.json_response({
                'error': str(e)
            }, status=500)
            
    async def start_server(self, port=8080):
        """Start API server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', port)
        await site.start()
        logger.info(f"Scaling API server started on port {port}")
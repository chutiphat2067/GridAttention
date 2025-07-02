"""
Augmentation Monitoring Dashboard for GridAttention
Real-time monitoring of phase-aware augmentation
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from collections import deque, defaultdict

import logging
logger = logging.getLogger(__name__)


class AugmentationMonitor:
    """Monitor and visualize augmentation statistics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Metrics storage
        self.augmentation_history = deque(maxlen=window_size)
        self.phase_transitions = []
        self.performance_correlation = deque(maxlen=100)
        
        # Real-time metrics
        self.current_metrics = {
            'total_augmented': 0,
            'by_phase': defaultdict(int),
            'by_method': defaultdict(int),
            'quality_scores': deque(maxlen=100),
            'augmentation_factors': deque(maxlen=100)
        }
        
        # Alerts
        self.alerts = deque(maxlen=50)
        
    async def update(self, augmentation_result: Dict[str, Any], 
                    performance_metrics: Dict[str, float],
                    current_phase: str):
        """Update monitoring with new augmentation data"""
        
        timestamp = datetime.now()
        
        # Record augmentation event
        if augmentation_result['augmentation_applied']:
            aug_info = augmentation_result.get('augmentation_info', {})
            
            event = {
                'timestamp': timestamp,
                'phase': current_phase,
                'augmentation_factor': aug_info.get('augmentation_factor', 1.0),
                'methods': aug_info.get('methods', []),
                'quality_score': aug_info.get('quality_score', 1.0),
                'performance': performance_metrics
            }
            
            self.augmentation_history.append(event)
            
            # Update counters
            self.current_metrics['total_augmented'] += augmentation_result['processed_count']
            self.current_metrics['by_phase'][current_phase] += augmentation_result['processed_count']
            
            for method in aug_info.get('methods', []):
                self.current_metrics['by_method'][method] += 1
                
            self.current_metrics['quality_scores'].append(aug_info.get('quality_score', 1.0))
            self.current_metrics['augmentation_factors'].append(aug_info.get('augmentation_factor', 1.0))
            
            # Check for anomalies
            await self._check_anomalies(event)
            
    async def _check_anomalies(self, event: Dict[str, Any]):
        """Check for augmentation anomalies"""
        
        # Alert if augmentation in active phase
        if event['phase'] == 'active':
            self.alerts.append({
                'timestamp': event['timestamp'],
                'type': 'ACTIVE_PHASE_AUGMENTATION',
                'severity': 'WARNING',
                'message': f"Augmentation applied in ACTIVE phase (factor: {event['augmentation_factor']:.2f})",
                'performance': event['performance']
            })
            
        # Alert if quality score too low
        if event['quality_score'] < 0.7:
            self.alerts.append({
                'timestamp': event['timestamp'],
                'type': 'LOW_QUALITY_AUGMENTATION',
                'severity': 'WARNING',
                'message': f"Low quality augmentation detected (score: {event['quality_score']:.2f})"
            })
            
        # Alert if augmentation factor too high
        if event['augmentation_factor'] > 5.0:
            self.alerts.append({
                'timestamp': event['timestamp'],
                'type': 'EXCESSIVE_AUGMENTATION',
                'severity': 'ERROR',
                'message': f"Excessive augmentation factor: {event['augmentation_factor']:.1f}x"
            })
            
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        
        # Calculate statistics
        recent_events = list(self.augmentation_history)
        
        if recent_events:
            avg_quality = np.mean([e['quality_score'] for e in recent_events])
            avg_factor = np.mean([e['augmentation_factor'] for e in recent_events])
            
            # Phase distribution
            phase_dist = defaultdict(int)
            for event in recent_events:
                phase_dist[event['phase']] += 1
                
            # Method usage
            method_usage = defaultdict(int)
            for event in recent_events:
                for method in event['methods']:
                    method_usage[method] += 1
                    
        else:
            avg_quality = 0
            avg_factor = 0
            phase_dist = {}
            method_usage = {}
            
        return {
            'summary': {
                'total_augmented': self.current_metrics['total_augmented'],
                'average_quality': avg_quality,
                'average_factor': avg_factor,
                'active_alerts': len([a for a in self.alerts if a['severity'] == 'ERROR'])
            },
            'phase_distribution': dict(phase_dist),
            'method_usage': dict(method_usage),
            'recent_alerts': list(self.alerts)[-10:],
            'time_series': {
                'quality_scores': list(self.current_metrics['quality_scores']),
                'augmentation_factors': list(self.current_metrics['augmentation_factors'])
            }
        }
        
    def generate_report(self) -> str:
        """Generate text report of augmentation statistics"""
        
        data = self.get_dashboard_data()
        
        report = f"""
=== Augmentation Monitoring Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Total Augmented Samples: {data['summary']['total_augmented']:,}
- Average Quality Score: {data['summary']['average_quality']:.3f}
- Average Augmentation Factor: {data['summary']['average_factor']:.2f}x
- Active Alerts: {data['summary']['active_alerts']}

PHASE DISTRIBUTION:
"""
        
        for phase, count in data['phase_distribution'].items():
            percentage = count / len(self.augmentation_history) * 100 if self.augmentation_history else 0
            report += f"- {phase.capitalize()}: {count} ({percentage:.1f}%)\n"
            
        report += "\nMETHOD USAGE:\n"
        for method, count in sorted(data['method_usage'].items(), key=lambda x: x[1], reverse=True):
            report += f"- {method}: {count}\n"
            
        if data['recent_alerts']:
            report += "\nRECENT ALERTS:\n"
            for alert in data['recent_alerts'][-5:]:
                report += f"- [{alert['severity']}] {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}\n"
                
        return report
        
    async def start_monitoring_loop(self, interval: int = 60):
        """Start monitoring loop that logs statistics"""
        while True:
            try:
                report = self.generate_report()
                logger.info(report)
                
                # Check for critical conditions
                if self.current_metrics['total_augmented'] > 0:
                    recent_quality = list(self.current_metrics['quality_scores'])
                    if recent_quality and np.mean(recent_quality) < 0.7:
                        logger.warning("⚠️ Average augmentation quality below threshold")
                        
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)


class AugmentationDashboard:
    """Web-based dashboard for augmentation monitoring"""
    
    def __init__(self, monitor: AugmentationMonitor):
        self.monitor = monitor
        self.dashboard_data = {}
        
    def get_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        data = self.monitor.get_dashboard_data()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GridAttention Augmentation Monitor</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
                .metric-label {{ color: #666; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
                .alert-WARNING {{ background: #fff3cd; border: 1px solid #ffeeba; }}
                .alert-ERROR {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
                .phase-bar {{ display: inline-block; padding: 5px 10px; margin: 2px; border-radius: 4px; }}
                .phase-learning {{ background: #e3f2fd; }}
                .phase-shadow {{ background: #fff9c4; }}
                .phase-active {{ background: #c8e6c9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 GridAttention Augmentation Monitor</h1>
                
                <div class="card">
                    <h2>Summary</h2>
                    <div class="metric">
                        <div class="metric-value">{data['summary']['total_augmented']:,}</div>
                        <div class="metric-label">Total Augmented</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{data['summary']['average_quality']:.3f}</div>
                        <div class="metric-label">Avg Quality</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{data['summary']['average_factor']:.2f}x</div>
                        <div class="metric-label">Avg Factor</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{data['summary']['active_alerts']}</div>
                        <div class="metric-label">Active Alerts</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Phase Distribution</h2>
        """
        
        for phase, count in data['phase_distribution'].items():
            percentage = count / sum(data['phase_distribution'].values()) * 100 if data['phase_distribution'] else 0
            html += f'<span class="phase-bar phase-{phase}">{phase}: {count} ({percentage:.1f}%)</span>'
            
        html += """
                </div>
                
                <div class="card">
                    <h2>Recent Alerts</h2>
        """
        
        for alert in data['recent_alerts'][-5:]:
            html += f"""
                    <div class="alert alert-{alert['severity']}">
                        <strong>{alert['timestamp'].strftime('%H:%M:%S')}</strong> - {alert['message']}
                    </div>
            """
            
        html += """
                </div>
            </div>
            <script>
                // Auto-refresh every 5 seconds
                setTimeout(() => location.reload(), 5000);
            </script>
        </body>
        </html>
        """
        
        return html


# Example usage
async def example_monitoring():
    """Example of using augmentation monitoring"""
    
    # Create monitor
    monitor = AugmentationMonitor()
    
    # Start monitoring loop
    monitor_task = asyncio.create_task(monitor.start_monitoring_loop(30))
    
    # Simulate augmentation events
    for i in range(100):
        # Simulate different phases
        if i < 40:
            phase = 'learning'
            aug_factor = 3.0
        elif i < 70:
            phase = 'shadow'
            aug_factor = 1.5
        else:
            phase = 'active'
            aug_factor = 1.0 if np.random.rand() > 0.9 else 0.0
            
        # Create augmentation result
        aug_result = {
            'augmentation_applied': aug_factor > 0,
            'processed_count': int(aug_factor) if aug_factor > 0 else 1,
            'augmentation_info': {
                'phase': phase,
                'augmentation_factor': aug_factor,
                'methods': ['noise_injection', 'bootstrap'] if aug_factor > 1 else [],
                'quality_score': 0.8 + np.random.rand() * 0.2
            }
        }
        
        # Update monitor
        await monitor.update(
            aug_result,
            {'win_rate': 0.5 + np.random.randn() * 0.1, 'sharpe_ratio': 1.0 + np.random.randn() * 0.5},
            phase
        )
        
        await asyncio.sleep(0.1)
        
    # Generate final report
    print(monitor.generate_report())
    
    # Create dashboard
    dashboard = AugmentationDashboard(monitor)
    html = dashboard.get_html_dashboard()
    
    # Save dashboard
    with open('augmentation_dashboard.html', 'w') as f:
        f.write(html)
    print("\nDashboard saved to augmentation_dashboard.html")
    
    # Cancel monitoring task
    monitor_task.cancel()


if __name__ == "__main__":
    asyncio.run(example_monitoring())
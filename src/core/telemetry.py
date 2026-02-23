import logging
import time
import json
from functools import wraps
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Telemetry:
    """Lightweight telemetry for tracking metrics and performance"""
    
    def __init__(self, component_name):
        self.logger = logging.getLogger(component_name)
        self.metrics = {}
    
    def log_info(self, message, **kwargs):
        """Log info with structured data"""
        extra = json.dumps(kwargs) if kwargs else ""
        self.logger.info(f"{message} {extra}")
    
    def log_error(self, message, error=None, **kwargs):
        """Log error with context"""
        kwargs['error'] = str(error) if error else None
        extra = json.dumps(kwargs)
        self.logger.error(f"{message} {extra}")
    
    def log_warning(self, message, **kwargs):
        """Log warning with context"""
        extra = json.dumps(kwargs) if kwargs else ""
        self.logger.warning(f"{message} {extra}")
    
    def track_metric(self, metric_name, value):
        """Track a metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
    
    def time_operation(self, operation_name):
        """Decorator to time operations"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start
                    self.log_info(f"{operation_name} completed", duration_ms=round(duration * 1000, 2))
                    self.track_metric(f"{operation_name}_duration", duration)
                    return result
                except Exception as e:
                    duration = time.time() - start
                    self.log_error(f"{operation_name} failed", error=e, duration_ms=round(duration * 1000, 2))
                    raise
            return wrapper
        return decorator

# Global telemetry instances
telemetry_instances = {}

def get_telemetry(component_name):
    """Get or create telemetry instance for a component"""
    if component_name not in telemetry_instances:
        telemetry_instances[component_name] = Telemetry(component_name)
    return telemetry_instances[component_name]

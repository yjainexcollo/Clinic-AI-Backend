"""
Azure Application Insights integration with OpenTelemetry and Live Metrics support
Migrated from OpenCensus to OpenTelemetry for better performance and Live Metrics
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Global flag to track initialization
_initialized = False


class AzureMonitorService:
    """Service for Azure Application Insights with Live Metrics support"""
    
    def __init__(self):
        self.connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        self.enabled = bool(self.connection_string)
        
        if self.enabled:
            try:
                # Only initialize once
                global _initialized
                if not _initialized:
                    self._setup_telemetry()
                    _initialized = True
                    logger.info("✅ Azure Application Insights telemetry enabled with Live Metrics")
                    logger.info("📊 Live Metrics: https://portal.azure.com → Application Insights → Live Metrics")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Azure Monitor: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️  Azure Application Insights not configured (APPLICATIONINSIGHTS_CONNECTION_STRING missing)")
    
    def _setup_telemetry(self):
        """Set up Azure Monitor with OpenTelemetry and Live Metrics"""
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            
            # Configure Azure Monitor with Live Metrics enabled
            # This automatically sets up traces, metrics, and logs
            configure_azure_monitor(
                connection_string=self.connection_string,
                enable_live_metrics=True,  # ✅ Enables Live Metrics with 1-second latency
                logger_name=__name__
            )
            
            logger.info("✅ Azure Monitor configured with OpenTelemetry")
            logger.info("✅ Live Metrics enabled - data will stream in real-time")
            
        except ImportError as e:
            logger.warning(f"⚠️  Azure Monitor OpenTelemetry dependencies not installed: {e}")
            logger.warning("💡 Install with: pip install azure-monitor-opentelemetry")
            self.enabled = False
        except Exception as e:
            logger.error(f"❌ Failed to set up Azure Monitor: {e}")
            self.enabled = False
    
    def track_event(self, name: str, properties: Optional[Dict[str, Any]] = None):
        """Track a custom event using OpenTelemetry"""
        if self.enabled:
            try:
                from opentelemetry import trace
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(name) as span:
                    if properties:
                        for key, value in properties.items():
                            span.set_attribute(str(key), str(value))
                    logger.info(f"CUSTOM_EVENT: {name}", extra=properties or {})
            except Exception as e:
                logger.warning(f"Failed to track event: {e}")
    
    def track_metric(self, name: str, value: float, properties: Optional[Dict[str, Any]] = None):
        """Track a custom metric using OpenTelemetry"""
        if self.enabled:
            try:
                from opentelemetry import metrics
                meter = metrics.get_meter(__name__)
                counter = meter.create_counter(name, description=f"Custom metric: {name}")
                counter.add(value, attributes=properties or {})
                logger.info(f"CUSTOM_METRIC: {name}={value}", extra=properties or {})
            except Exception as e:
                logger.warning(f"Failed to track metric: {e}")


# Singleton instance
_monitor_service: Optional[AzureMonitorService] = None


def get_azure_monitor() -> AzureMonitorService:
    """Get Azure Monitor service instance"""
    global _monitor_service
    if _monitor_service is None:
        _monitor_service = AzureMonitorService()
    return _monitor_service


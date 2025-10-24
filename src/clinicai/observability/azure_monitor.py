"""
Azure Application Insights integration for custom telemetry and Live Metrics
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Global flag to track initialization
_initialized = False


class AzureMonitorService:
    """Service for Azure Application Insights custom telemetry"""
    
    def __init__(self):
        self.connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        self.enabled = bool(self.connection_string)
        self.instrumentation_key = None
        
        if self.enabled:
            try:
                # Extract instrumentation key from connection string
                for part in self.connection_string.split(';'):
                    if part.startswith('InstrumentationKey='):
                        self.instrumentation_key = part.split('=')[1]
                        break
                
                # Only initialize once
                global _initialized
                if not _initialized and self.instrumentation_key:
                    self._setup_telemetry()
                    _initialized = True
                    logger.info("✅ Azure Application Insights telemetry enabled")
                    logger.info(f"📊 Instrumentation Key: {self.instrumentation_key[:8]}...")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Azure Monitor: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️  Azure Application Insights not configured (APPLICATIONINSIGHTS_CONNECTION_STRING missing)")
    
    def _setup_telemetry(self):
        """Set up Azure Monitor exporters"""
        try:
            from opencensus.ext.azure.log_exporter import AzureLogHandler
            from opencensus.ext.azure.trace_exporter import AzureExporter
            from opencensus.trace import config_integration
            from opencensus.trace.samplers import ProbabilitySampler
            from opencensus.trace.tracer import Tracer
            
            # Enable integrations for automatic instrumentation
            config_integration.trace_integrations(['logging', 'requests', 'httplib'])
            
            # Add Azure Log Handler to root logger
            azure_handler = AzureLogHandler(connection_string=self.connection_string)
            logging.getLogger().addHandler(azure_handler)
            
            # Set up tracing for Live Metrics
            tracer = Tracer(
                exporter=AzureExporter(connection_string=self.connection_string),
                sampler=ProbabilitySampler(1.0)  # Sample 100% of requests
            )
            
            logger.info("✅ Azure Monitor exporters configured")
            
        except ImportError as e:
            logger.warning(f"⚠️  Azure Monitor dependencies not fully installed: {e}")
            logger.warning("💡 Install with: pip install opencensus-ext-azure opencensus-ext-requests")
        except Exception as e:
            logger.error(f"❌ Failed to set up Azure Monitor: {e}")
    
    def track_event(self, name: str, properties: Optional[Dict[str, Any]] = None):
        """Track a custom event"""
        if self.enabled:
            logger.info(f"CUSTOM_EVENT: {name}", extra={
                'custom_dimensions': properties or {}
            })
    
    def track_metric(self, name: str, value: float, properties: Optional[Dict[str, Any]] = None):
        """Track a custom metric"""
        if self.enabled:
            logger.info(f"CUSTOM_METRIC: {name}={value}", extra={
                'custom_dimensions': properties or {}
            })


# Singleton instance
_monitor_service: Optional[AzureMonitorService] = None


def get_azure_monitor() -> AzureMonitorService:
    """Get Azure Monitor service instance"""
    global _monitor_service
    if _monitor_service is None:
        _monitor_service = AzureMonitorService()
    return _monitor_service


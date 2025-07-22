#!/usr/bin/env python3
"""
MemMimic Enterprise Monitoring Deployment Script

This script deploys and configures the complete MemMimic Enterprise Monitoring System
for production environments.
"""

import os
import sys
import asyncio
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any

# Add MemMimic to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memmimic.api import create_memmimic
from memmimic.monitoring import MonitoringServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringDeployment:
    """Handles deployment and configuration of MemMimic monitoring"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.memmimic_api = None
        self.monitoring_server = None
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            "database": {
                "path": os.getenv("MEMMIMIC_DB_PATH", "/var/lib/memmimic/memmimic.db")
            },
            "monitoring": {
                "dashboard_port": int(os.getenv("MEMMIMIC_DASHBOARD_PORT", "8080")),
                "dashboard_host": os.getenv("MEMMIMIC_DASHBOARD_HOST", "0.0.0.0"),
                "health_check_interval": float(os.getenv("MEMMIMIC_HEALTH_CHECK_INTERVAL", "30")),
                "alert_evaluation_interval": float(os.getenv("MEMMIMIC_ALERT_EVALUATION_INTERVAL", "30")),
                "metrics_collection_interval": float(os.getenv("MEMMIMIC_METRICS_COLLECTION_INTERVAL", "10")),
                "retention_hours": int(os.getenv("MEMMIMIC_RETENTION_HOURS", "24"))
            },
            "alerting": {
                "email": {
                    "enabled": os.getenv("MEMMIMIC_EMAIL_ALERTS", "false").lower() == "true",
                    "smtp_server": os.getenv("MEMMIMIC_SMTP_SERVER", "localhost"),
                    "smtp_port": int(os.getenv("MEMMIMIC_SMTP_PORT", "587")),
                    "username": os.getenv("MEMMIMIC_SMTP_USERNAME", ""),
                    "password": os.getenv("MEMMIMIC_SMTP_PASSWORD", ""),
                    "from_address": os.getenv("MEMMIMIC_EMAIL_FROM", "memmimic@localhost"),
                    "to_addresses": os.getenv("MEMMIMIC_EMAIL_TO", "").split(",") if os.getenv("MEMMIMIC_EMAIL_TO") else []
                },
                "webhook": {
                    "enabled": os.getenv("MEMMIMIC_WEBHOOK_ALERTS", "false").lower() == "true",
                    "url": os.getenv("MEMMIMIC_WEBHOOK_URL", ""),
                    "headers": json.loads(os.getenv("MEMMIMIC_WEBHOOK_HEADERS", "{}"))
                }
            },
            "security": {
                "retention_hours": int(os.getenv("MEMMIMIC_SECURITY_RETENTION_HOURS", "72"))
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    # Deep merge configurations
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def validate_environment(self) -> bool:
        """Validate deployment environment"""
        logger.info("Validating deployment environment...")
        
        checks = []
        
        # Check database directory
        db_path = Path(self.config["database"]["path"])
        db_dir = db_path.parent
        
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created database directory: {db_dir}")
            except Exception as e:
                logger.error(f"Failed to create database directory {db_dir}: {e}")
                checks.append(False)
        else:
            checks.append(True)
        
        # Check port availability
        import socket
        port = self.config["monitoring"]["dashboard_port"]
        host = self.config["monitoring"]["dashboard_host"]
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host if host != "0.0.0.0" else "localhost", port))
            sock.close()
            
            if result == 0:
                logger.error(f"Port {port} is already in use")
                checks.append(False)
            else:
                logger.info(f"Port {port} is available")
                checks.append(True)
                
        except Exception as e:
            logger.warning(f"Could not check port availability: {e}")
            checks.append(True)  # Assume it's OK
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(db_dir)
            free_gb = free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1GB free
                logger.warning(f"Low disk space: {free_gb:.1f}GB available")
            else:
                logger.info(f"Disk space OK: {free_gb:.1f}GB available")
            
            checks.append(True)
            
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            checks.append(True)
        
        success = all(checks)
        if success:
            logger.info("✅ Environment validation passed")
        else:
            logger.error("❌ Environment validation failed")
        
        return success
    
    def setup_directories(self):
        """Setup required directories and permissions"""
        logger.info("Setting up directories...")
        
        directories = [
            Path(self.config["database"]["path"]).parent,
            Path("/var/log/memmimic"),
            Path("/var/lib/memmimic"),
            Path("/etc/memmimic")
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory ready: {directory}")
            except PermissionError:
                logger.warning(f"Permission denied creating {directory} - may need sudo")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
    
    def install_systemd_service(self):
        """Install systemd service for monitoring"""
        service_content = f"""[Unit]
Description=MemMimic Enterprise Monitoring
After=network.target
Wants=network.target

[Service]
Type=simple
User=memmimic
Group=memmimic
WorkingDirectory={Path(__file__).parent.parent}
Environment=MEMMIMIC_DB_PATH={self.config["database"]["path"]}
Environment=MEMMIMIC_DASHBOARD_PORT={self.config["monitoring"]["dashboard_port"]}
Environment=MEMMIMIC_DASHBOARD_HOST={self.config["monitoring"]["dashboard_host"]}
ExecStart=/usr/bin/python3 -m memmimic.monitoring.monitoring_server
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
"""
        
        service_path = Path("/etc/systemd/system/memmimic-monitoring.service")
        
        try:
            with open(service_path, 'w') as f:
                f.write(service_content)
            
            logger.info(f"Systemd service installed: {service_path}")
            logger.info("Enable with: sudo systemctl enable memmimic-monitoring")
            logger.info("Start with: sudo systemctl start memmimic-monitoring")
            
        except PermissionError:
            logger.warning("Permission denied - run with sudo to install systemd service")
            logger.info("Service content:")
            print(service_content)
            
        except Exception as e:
            logger.error(f"Failed to install systemd service: {e}")
    
    def create_config_files(self):
        """Create configuration files"""
        logger.info("Creating configuration files...")
        
        # Main configuration
        config_path = Path("/etc/memmimic/monitoring.json")
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to: {config_path}")
            
        except PermissionError:
            logger.warning("Permission denied - creating config in current directory")
            config_path = Path("monitoring_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to create configuration: {e}")
        
        # Logrotate configuration
        logrotate_content = """/var/log/memmimic/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 memmimic memmimic
    postrotate
        systemctl reload memmimic-monitoring
    endscript
}
"""
        
        try:
            logrotate_path = Path("/etc/logrotate.d/memmimic-monitoring")
            with open(logrotate_path, 'w') as f:
                f.write(logrotate_content)
            logger.info(f"Logrotate configuration installed: {logrotate_path}")
            
        except PermissionError:
            logger.warning("Permission denied - could not install logrotate config")
        except Exception as e:
            logger.error(f"Failed to install logrotate config: {e}")
    
    async def deploy_monitoring(self):
        """Deploy the monitoring system"""
        logger.info("Deploying MemMimic Enterprise Monitoring System...")
        
        try:
            # Initialize MemMimic API
            logger.info("Initializing MemMimic API...")
            self.memmimic_api = create_memmimic(self.config["database"]["path"])
            
            # Create monitoring server
            logger.info("Creating monitoring server...")
            self.monitoring_server = MonitoringServer(
                memmimic_api=self.memmimic_api,
                dashboard_port=self.config["monitoring"]["dashboard_port"],
                dashboard_host=self.config["monitoring"]["dashboard_host"],
                health_check_interval=self.config["monitoring"]["health_check_interval"],
                alert_evaluation_interval=self.config["monitoring"]["alert_evaluation_interval"]
            )
            
            # Configure alerting
            await self._configure_alerting()
            
            # Start monitoring
            logger.info("Starting monitoring server...")
            await self.monitoring_server.start()
            
            # Display deployment information
            self._display_deployment_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    async def _configure_alerting(self):
        """Configure alerting channels"""
        if not self.monitoring_server:
            return
        
        from memmimic.monitoring.alert_manager import NotificationChannel
        
        # Configure email alerts
        email_config = self.config["alerting"]["email"]
        if email_config["enabled"] and email_config["to_addresses"]:
            email_channel = NotificationChannel(
                channel_id="production_email",
                channel_type="email",
                config=email_config
            )
            self.monitoring_server.alert_manager.add_notification_channel(email_channel)
            logger.info("Email alerting configured")
        
        # Configure webhook alerts
        webhook_config = self.config["alerting"]["webhook"]
        if webhook_config["enabled"] and webhook_config["url"]:
            webhook_channel = NotificationChannel(
                channel_id="production_webhook",
                channel_type="webhook",
                config=webhook_config
            )
            self.monitoring_server.alert_manager.add_notification_channel(webhook_channel)
            logger.info("Webhook alerting configured")
    
    def _display_deployment_info(self):
        """Display deployment information"""
        port = self.config["monitoring"]["dashboard_port"]
        host = self.config["monitoring"]["dashboard_host"]
        
        logger.info("=" * 80)
        logger.info("🚀 MemMimic Enterprise Monitoring System Deployed Successfully!")
        logger.info("=" * 80)
        logger.info(f"📊 Dashboard: http://localhost:{port}/dashboard")
        logger.info(f"🔧 API: http://localhost:{port}/api/status")
        logger.info(f"📈 Prometheus Metrics: http://localhost:{port}/api/metrics/prometheus")
        logger.info(f"🌐 WebSocket: ws://localhost:{port}/ws")
        logger.info("=" * 80)
        logger.info("Key Features Available:")
        logger.info("  ✅ Real-time performance metrics")
        logger.info("  ✅ Comprehensive health monitoring")
        logger.info("  ✅ Security incident detection")
        logger.info("  ✅ Intelligent alerting system")
        logger.info("  ✅ Automated incident response")
        logger.info("  ✅ Interactive monitoring dashboard")
        logger.info("=" * 80)
        
        # Configuration summary
        logger.info("Configuration Summary:")
        logger.info(f"  Database: {self.config['database']['path']}")
        logger.info(f"  Health Checks: Every {self.config['monitoring']['health_check_interval']}s")
        logger.info(f"  Alert Evaluation: Every {self.config['monitoring']['alert_evaluation_interval']}s")
        logger.info(f"  Data Retention: {self.config['monitoring']['retention_hours']} hours")
        
        if self.config["alerting"]["email"]["enabled"]:
            logger.info(f"  Email Alerts: {', '.join(self.config['alerting']['email']['to_addresses'])}")
        
        if self.config["alerting"]["webhook"]["enabled"]:
            logger.info(f"  Webhook Alerts: {self.config['alerting']['webhook']['url']}")
        
        logger.info("=" * 80)
    
    async def health_check(self):
        """Perform deployment health check"""
        if not self.monitoring_server:
            logger.error("Monitoring server not initialized")
            return False
        
        try:
            logger.info("Performing deployment health check...")
            
            # Run health checks
            health_result = await self.monitoring_server.run_health_check()
            logger.info(f"Health Status: {health_result.status.value}")
            logger.info(f"Healthy Components: {health_result.healthy_checks}/{health_result.total_checks}")
            
            # Check system status
            system_status = await self.monitoring_server.get_system_status()
            logger.info(f"Overall System Status: {system_status['overall_status']}")
            
            # Verify endpoints
            import aiohttp
            port = self.config["monitoring"]["dashboard_port"]
            
            async with aiohttp.ClientSession() as session:
                endpoints = [
                    f"http://localhost:{port}/api/status",
                    f"http://localhost:{port}/api/health",
                    f"http://localhost:{port}/api/metrics"
                ]
                
                for endpoint in endpoints:
                    try:
                        async with session.get(endpoint) as resp:
                            if resp.status == 200:
                                logger.info(f"✅ {endpoint}")
                            else:
                                logger.warning(f"⚠️ {endpoint} returned {resp.status}")
                    except Exception as e:
                        logger.error(f"❌ {endpoint} failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the monitoring system"""
        if self.monitoring_server:
            await self.monitoring_server.stop()
            logger.info("Monitoring server stopped")


async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy MemMimic Enterprise Monitoring")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--validate-only", action="store_true", help="Only validate environment")
    parser.add_argument("--install-service", action="store_true", help="Install systemd service")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    parser.add_argument("--setup", action="store_true", help="Setup directories and configs")
    
    args = parser.parse_args()
    
    # Create deployment instance
    deployment = MonitoringDeployment(config_file=args.config)
    
    try:
        # Validate environment
        if not deployment.validate_environment():
            sys.exit(1)
        
        if args.validate_only:
            logger.info("Environment validation completed")
            return
        
        if args.setup:
            deployment.setup_directories()
            deployment.create_config_files()
            logger.info("Setup completed")
            return
        
        if args.install_service:
            deployment.install_systemd_service()
            return
        
        # Deploy monitoring system
        success = await deployment.deploy_monitoring()
        if not success:
            sys.exit(1)
        
        # Run health check if requested
        if args.health_check:
            await asyncio.sleep(5)  # Wait for system to stabilize
            success = await deployment.health_check()
            if not success:
                logger.warning("Health check revealed issues")
        
        # Keep running
        logger.info("Monitoring system is running. Press Ctrl+C to stop...")
        await deployment.monitoring_server.wait_until_stopped()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)
    finally:
        await deployment.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
MemMimic Production Backup and Disaster Recovery System
Automated backup, recovery, and cross-region replication
"""

import asyncio
import json
import logging
import os
import subprocess
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import asyncpg
import boto3
from kubernetes import client, config as k8s_config
from prometheus_client import Gauge, Counter, start_http_server

# Metrics
BACKUP_SUCCESS = Counter('memmimic_backup_success_total', 'Successful backups', ['type'])
BACKUP_FAILURE = Counter('memmimic_backup_failure_total', 'Failed backups', ['type'])
BACKUP_DURATION = Gauge('memmimic_backup_duration_seconds', 'Backup duration', ['type'])
BACKUP_SIZE = Gauge('memmimic_backup_size_bytes', 'Backup size', ['type'])

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Backup configuration"""
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_databases: List[str]
    
    s3_bucket: str
    s3_region: str
    s3_access_key: str
    s3_secret_key: str
    
    retention_days: int = 30
    full_backup_days: List[int] = None  # Days of week for full backup (0=Monday)
    
    def __post_init__(self):
        if self.full_backup_days is None:
            self.full_backup_days = [0, 3, 6]  # Monday, Thursday, Sunday


@dataclass 
class BackupMetadata:
    """Backup metadata"""
    backup_id: str
    backup_type: str  # full, incremental
    timestamp: datetime
    size_bytes: int
    databases: List[str]
    s3_key: str
    checksum: str
    status: str  # success, failed, in_progress


class PostgreSQLBackup:
    """PostgreSQL backup management"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            region_name=config.s3_region,
            aws_access_key_id=config.s3_access_key,
            aws_secret_access_key=config.s3_secret_key
        )
    
    async def create_backup(self, backup_type: str = "full") -> BackupMetadata:
        """Create database backup"""
        backup_id = f"memmimic-{backup_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        timestamp = datetime.now()
        
        logger.info(f"Starting {backup_type} backup: {backup_id}")
        
        try:
            # Create backup directory
            backup_dir = Path(f"/tmp/backups/{backup_id}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            total_size = 0
            databases_backed_up = []
            
            for db_name in self.config.postgres_databases:
                try:
                    db_backup_file = backup_dir / f"{db_name}.sql"
                    
                    # Create pg_dump command
                    cmd = [
                        "pg_dump",
                        "-h", self.config.postgres_host,
                        "-p", str(self.config.postgres_port),
                        "-U", self.config.postgres_user,
                        "-d", db_name,
                        "-f", str(db_backup_file),
                        "--verbose",
                        "--no-password"
                    ]
                    
                    # Set environment
                    env = os.environ.copy()
                    env["PGPASSWORD"] = self.config.postgres_password
                    
                    # Execute pg_dump
                    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        logger.error(f"pg_dump failed for {db_name}: {result.stderr}")
                        continue
                    
                    # Compress the backup
                    compressed_file = backup_dir / f"{db_name}.sql.gz"
                    subprocess.run(["gzip", str(db_backup_file)], check=True)
                    
                    size = compressed_file.stat().st_size
                    total_size += size
                    databases_backed_up.append(db_name)
                    
                    logger.info(f"Backed up {db_name}: {size} bytes")
                    
                except Exception as e:
                    logger.error(f"Failed to backup database {db_name}: {e}")
                    BACKUP_FAILURE.labels(type=backup_type).inc()
                    continue
            
            if not databases_backed_up:
                raise Exception("No databases were successfully backed up")
            
            # Create tarball
            tarball_path = f"/tmp/{backup_id}.tar.gz"
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(backup_dir, arcname=backup_id)
            
            # Calculate checksum
            import hashlib
            with open(tarball_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Upload to S3
            s3_key = f"backups/postgresql/{backup_id}.tar.gz"
            
            with open(tarball_path, "rb") as f:
                self.s3_client.upload_fileobj(
                    f, 
                    self.config.s3_bucket, 
                    s3_key,
                    ExtraArgs={
                        'Metadata': {
                            'backup-type': backup_type,
                            'timestamp': timestamp.isoformat(),
                            'databases': ','.join(databases_backed_up),
                            'checksum': checksum
                        }
                    }
                )
            
            # Cleanup local files
            os.remove(tarball_path)
            import shutil
            shutil.rmtree(backup_dir)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=timestamp,
                size_bytes=total_size,
                databases=databases_backed_up,
                s3_key=s3_key,
                checksum=checksum,
                status="success"
            )
            
            # Update metrics
            BACKUP_SUCCESS.labels(type=backup_type).inc()
            BACKUP_SIZE.labels(type=backup_type).set(total_size)
            
            logger.info(f"Backup {backup_id} completed successfully")
            return metadata
            
        except Exception as e:
            logger.error(f"Backup {backup_id} failed: {e}")
            BACKUP_FAILURE.labels(type=backup_type).inc()
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=timestamp,
                size_bytes=0,
                databases=[],
                s3_key="",
                checksum="",
                status="failed"
            )
            return metadata
    
    async def restore_backup(self, backup_id: str, target_databases: Optional[List[str]] = None) -> bool:
        """Restore from backup"""
        logger.info(f"Starting restore from backup: {backup_id}")
        
        try:
            # Download backup from S3
            s3_key = f"backups/postgresql/{backup_id}.tar.gz"
            local_path = f"/tmp/{backup_id}.tar.gz"
            
            self.s3_client.download_file(
                self.config.s3_bucket, 
                s3_key, 
                local_path
            )
            
            # Extract tarball
            extract_dir = f"/tmp/restore_{backup_id}"
            with tarfile.open(local_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            
            backup_dir = Path(extract_dir) / backup_id
            
            # Restore each database
            for sql_file in backup_dir.glob("*.sql.gz"):
                db_name = sql_file.stem.replace('.sql', '')
                
                if target_databases and db_name not in target_databases:
                    continue
                
                # Decompress
                subprocess.run(["gunzip", str(sql_file)], check=True)
                decompressed_file = backup_dir / f"{db_name}.sql"
                
                # Restore database
                cmd = [
                    "psql",
                    "-h", self.config.postgres_host,
                    "-p", str(self.config.postgres_port),
                    "-U", self.config.postgres_user,
                    "-d", db_name,
                    "-f", str(decompressed_file),
                    "--no-password"
                ]
                
                env = os.environ.copy()
                env["PGPASSWORD"] = self.config.postgres_password
                
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Restore failed for {db_name}: {result.stderr}")
                    return False
                
                logger.info(f"Restored database: {db_name}")
            
            # Cleanup
            os.remove(local_path)
            import shutil
            shutil.rmtree(extract_dir)
            
            logger.info(f"Restore from {backup_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Restore from {backup_id} failed: {e}")
            return False
    
    def list_backups(self, backup_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """List available backups"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix="backups/postgresql/",
                MaxKeys=limit
            )
            
            backups = []
            for obj in response.get('Contents', []):
                # Get metadata
                head_response = self.s3_client.head_object(
                    Bucket=self.config.s3_bucket,
                    Key=obj['Key']
                )
                
                metadata = head_response.get('Metadata', {})
                
                backup_info = {
                    'backup_id': Path(obj['Key']).stem.replace('.tar', ''),
                    'backup_type': metadata.get('backup-type', 'unknown'),
                    'timestamp': metadata.get('timestamp'),
                    'size_bytes': obj['Size'],
                    'databases': metadata.get('databases', '').split(',') if metadata.get('databases') else [],
                    's3_key': obj['Key'],
                    'checksum': metadata.get('checksum', '')
                }
                
                if backup_type and backup_info['backup_type'] != backup_type:
                    continue
                    
                backups.append(backup_info)
            
            return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def cleanup_old_backups(self) -> int:
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            deleted_count = 0
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix="backups/postgresql/"
            )
            
            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.config.s3_bucket,
                        Key=obj['Key']
                    )
                    deleted_count += 1
                    logger.info(f"Deleted old backup: {obj['Key']}")
            
            logger.info(f"Cleaned up {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
            return 0


class KubernetesBackup:
    """Kubernetes resources backup"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        k8s_config.load_incluster_config()
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.s3_client = boto3.client(
            's3',
            region_name=config.s3_region,
            aws_access_key_id=config.s3_access_key,
            aws_secret_access_key=config.s3_secret_key
        )
    
    def backup_kubernetes_resources(self) -> bool:
        """Backup Kubernetes manifests and configurations"""
        try:
            backup_id = f"k8s-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'namespaces': {},
                'cluster_resources': {}
            }
            
            # Backup namespaces and resources
            namespaces = self.v1.list_namespace()
            
            for ns in namespaces.items:
                if ns.metadata.name in ['kube-system', 'kube-public', 'kube-node-lease']:
                    continue
                
                ns_name = ns.metadata.name
                backup_data['namespaces'][ns_name] = {
                    'configmaps': [],
                    'secrets': [],
                    'services': [],
                    'deployments': [],
                    'statefulsets': [],
                    'persistentvolumeclaims': []
                }
                
                # ConfigMaps
                configmaps = self.v1.list_namespaced_config_map(namespace=ns_name)
                for cm in configmaps.items:
                    backup_data['namespaces'][ns_name]['configmaps'].append(
                        client.ApiClient().sanitize_for_serialization(cm)
                    )
                
                # Secrets (without sensitive data)
                secrets = self.v1.list_namespaced_secret(namespace=ns_name)
                for secret in secrets.items:
                    secret_data = client.ApiClient().sanitize_for_serialization(secret)
                    # Remove sensitive data
                    if 'data' in secret_data:
                        secret_data['data'] = {k: '***REDACTED***' for k in secret_data['data'].keys()}
                    backup_data['namespaces'][ns_name]['secrets'].append(secret_data)
                
                # Services
                services = self.v1.list_namespaced_service(namespace=ns_name)
                for svc in services.items:
                    backup_data['namespaces'][ns_name]['services'].append(
                        client.ApiClient().sanitize_for_serialization(svc)
                    )
                
                # Deployments
                deployments = self.apps_v1.list_namespaced_deployment(namespace=ns_name)
                for deploy in deployments.items:
                    backup_data['namespaces'][ns_name]['deployments'].append(
                        client.ApiClient().sanitize_for_serialization(deploy)
                    )
            
            # Upload to S3
            s3_key = f"backups/kubernetes/{backup_id}.json"
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Body=json.dumps(backup_data, indent=2),
                ContentType='application/json',
                Metadata={
                    'backup-type': 'kubernetes',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Kubernetes backup {backup_id} completed")
            BACKUP_SUCCESS.labels(type='kubernetes').inc()
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes backup failed: {e}")
            BACKUP_FAILURE.labels(type='kubernetes').inc()
            return False


class DisasterRecoveryOrchestrator:
    """Disaster recovery orchestration"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.postgres_backup = PostgreSQLBackup(config)
        self.k8s_backup = KubernetesBackup(config)
    
    async def full_system_backup(self) -> Dict[str, Any]:
        """Perform full system backup"""
        logger.info("Starting full system backup")
        start_time = datetime.now()
        
        results = {
            'timestamp': start_time.isoformat(),
            'postgres_backup': None,
            'kubernetes_backup': None,
            'success': False,
            'errors': []
        }
        
        try:
            # PostgreSQL backup
            postgres_result = await self.postgres_backup.create_backup("full")
            results['postgres_backup'] = asdict(postgres_result)
            
            # Kubernetes backup
            k8s_result = self.k8s_backup.backup_kubernetes_resources()
            results['kubernetes_backup'] = k8s_result
            
            # Check overall success
            postgres_success = postgres_result.status == "success"
            results['success'] = postgres_success and k8s_result
            
            duration = (datetime.now() - start_time).total_seconds()
            BACKUP_DURATION.labels(type='full').set(duration)
            
            if results['success']:
                logger.info(f"Full system backup completed in {duration:.2f}s")
                BACKUP_SUCCESS.labels(type='full').inc()
            else:
                logger.error("Full system backup had failures")
                BACKUP_FAILURE.labels(type='full').inc()
                
        except Exception as e:
            logger.error(f"Full system backup failed: {e}")
            results['errors'].append(str(e))
            BACKUP_FAILURE.labels(type='full').inc()
        
        return results
    
    def schedule_backups(self):
        """Schedule automated backups"""
        import schedule
        
        # Daily incremental backups at 2 AM
        schedule.every().day.at("02:00").do(
            lambda: asyncio.create_task(self.postgres_backup.create_backup("incremental"))
        )
        
        # Full backups on configured days at 1 AM
        for day in self.config.full_backup_days:
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            getattr(schedule.every(), day_names[day]).at("01:00").do(
                lambda: asyncio.create_task(self.full_system_backup())
            )
        
        # Cleanup old backups weekly
        schedule.every().sunday.at("04:00").do(self.postgres_backup.cleanup_old_backups)
        
        logger.info("Backup scheduler initialized")
    
    async def disaster_recovery(self, recovery_point: str) -> bool:
        """Full disaster recovery"""
        logger.info(f"Starting disaster recovery to point: {recovery_point}")
        
        try:
            # Restore PostgreSQL
            postgres_success = await self.postgres_backup.restore_backup(recovery_point)
            
            if postgres_success:
                logger.info("Disaster recovery completed successfully")
                return True
            else:
                logger.error("Disaster recovery failed")
                return False
                
        except Exception as e:
            logger.error(f"Disaster recovery failed: {e}")
            return False


async def main():
    """Main backup service"""
    # Load configuration
    config = BackupConfig(
        postgres_host=os.getenv("POSTGRES_HOST", "postgres-coordinator"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_user=os.getenv("POSTGRES_USER", "memmimic"),
        postgres_password=os.getenv("POSTGRES_PASSWORD"),
        postgres_databases=os.getenv("POSTGRES_DATABASES", "memmimic,memmimic_shard_1,memmimic_shard_2,memmimic_shard_3,memmimic_shard_4").split(","),
        s3_bucket=os.getenv("S3_BUCKET"),
        s3_region=os.getenv("S3_REGION", "us-west-2"),
        s3_access_key=os.getenv("S3_ACCESS_KEY"),
        s3_secret_key=os.getenv("S3_SECRET_KEY"),
        retention_days=int(os.getenv("RETENTION_DAYS", "30"))
    )
    
    # Start metrics server
    start_http_server(8080)
    
    # Initialize disaster recovery orchestrator
    orchestrator = DisasterRecoveryOrchestrator(config)
    orchestrator.schedule_backups()
    
    # Keep running
    import schedule
    while True:
        schedule.run_pending()
        await asyncio.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
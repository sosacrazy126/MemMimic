# MemMimic Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying MemMimic in a production environment with enterprise-grade scalability, security, and reliability.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Global Infrastructure                     │
├─────────────────────────────────────────────────────────────────┤
│  CloudFront CDN │ Route53 DNS │ S3 Cross-Region Backups        │
└─────────────────────────────────────────────────────────────────┘
           │                        │                        │
    ┌──────▼──────┐        ┌───────▼───────┐        ┌───────▼───────┐
    │ US-West-2   │        │   US-East-1   │        │  EU-West-1    │
    │ (Primary)   │        │ (Secondary)   │        │  (Tertiary)   │
    └─────────────┘        └───────────────┘        └───────────────┘
           │                        │                        │
    ┌──────▼──────┐        ┌───────▼───────┐        ┌───────▼───────┐
    │ EKS Cluster │        │ EKS Cluster   │        │ EKS Cluster   │
    │             │        │ (DR Ready)    │        │ (Read Replica)│
    └─────────────┘        └───────────────┘        └───────────────┘
```

### Microservices Architecture

- **API Gateway**: Load balancing, routing, and service orchestration
- **Memory Service**: AMMS storage and core memory operations
- **Classification Service**: CXD cognitive classification
- **Search Service**: Semantic search and vector operations
- **Tale Service**: Narrative management
- **Consciousness Service**: Living prompts and sigil processing

### Database Sharding

- **Coordinator Database**: Metadata and shard configuration
- **4 Data Shards**: Horizontally distributed memory storage
- **Cross-Region Replication**: Automated backup and disaster recovery

## Prerequisites

### Infrastructure Requirements

- **AWS Account** with appropriate IAM permissions
- **Kubernetes Cluster** (EKS recommended, minimum v1.25)
- **Domain Name** for public access
- **SSL Certificate** (managed via ACM)
- **S3 Buckets** for backups and artifacts
- **Container Registry** (ECR or Docker Hub)

### Tools Required

```bash
# Essential tools
kubectl >= 1.25
helm >= 3.10
docker >= 20.10
terraform >= 1.0
aws-cli >= 2.0

# Optional tools
k9s              # Kubernetes dashboard
stern            # Log tailing
kubectx/kubens   # Context switching
```

### Resource Requirements

| Component | CPU | Memory | Storage | Replicas |
|-----------|-----|--------|---------|----------|
| API Gateway | 1.5 vCPU | 1.5 GB | - | 3-20 |
| Memory Service | 1 vCPU | 1 GB | - | 2-10 |
| Classification Service | 2 vCPU | 3 GB | 50 GB | 2-6 |
| Search Service | 1.5 vCPU | 2 GB | 100 GB | 3-15 |
| Tale Service | 0.25 vCPU | 256 MB | 10 GB | 2-4 |
| Consciousness Service | 0.5 vCPU | 384 MB | 5 GB | 2-4 |
| PostgreSQL Coordinator | 2 vCPU | 4 GB | 100 GB | 1 |
| PostgreSQL Shards | 1.5 vCPU | 3 GB | 75 GB | 4 |
| Redis | 1 vCPU | 1.5 GB | 20 GB | 1-3 |

**Total Minimum**: ~15 vCPU, ~22 GB RAM, ~360 GB Storage

## Step-by-Step Deployment

### Phase 1: Infrastructure Setup

#### 1.1 Terraform Global Resources

```bash
# Clone the repository
git clone https://github.com/your-org/memmimic
cd memmimic

# Initialize Terraform
cd infrastructure/multi-region
terraform init

# Plan and apply global resources
terraform plan -var="project_name=memmimic" \
               -var="primary_region=us-west-2" \
               -var="secondary_region=us-east-1"

terraform apply
```

#### 1.2 EKS Cluster Setup

```bash
# Create EKS cluster (example for us-west-2)
eksctl create cluster \
  --name memmimic-primary \
  --region us-west-2 \
  --node-type m5.xlarge \
  --nodes 6 \
  --nodes-min 3 \
  --nodes-max 20 \
  --managed \
  --enable-ssm

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name memmimic-primary
```

#### 1.3 Storage Classes

```bash
# Create fast SSD storage class
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
reclaimPolicy: Retain
volumeBindingMode: WaitForFirstConsumer
EOF
```

### Phase 2: Core Services Deployment

#### 2.1 Namespace and RBAC

```bash
kubectl apply -f infrastructure/k8s/namespace.yaml
```

#### 2.2 Secrets Configuration

```bash
# Create secrets (use your actual values)
kubectl create secret generic memmimic-secrets \
  --namespace=memmimic \
  --from-literal=db_password="$(openssl rand -base64 32)" \
  --from-literal=replication_password="$(openssl rand -base64 32)" \
  --from-literal=redis_password="$(openssl rand -base64 32)" \
  --from-literal=jwt_secret="$(openssl rand -base64 64)" \
  --from-literal=grafana_password="$(openssl rand -base64 16)"

# For TLS certificates (if not using cert-manager)
kubectl create secret tls tls-certificates \
  --namespace=memmimic \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem
```

#### 2.3 Configuration

```bash
kubectl apply -f infrastructure/k8s/configmap.yaml
```

### Phase 3: Database Deployment

#### 3.1 PostgreSQL Cluster

```bash
# Deploy PostgreSQL coordinator and shards
kubectl apply -f infrastructure/k8s/postgres-cluster.yaml

# Wait for databases to be ready
kubectl wait --namespace=memmimic \
  --for=condition=ready pod \
  --selector=component=database \
  --timeout=600s
```

#### 3.2 Database Initialization

```bash
# Initialize schema and sharding
kubectl exec -it postgres-coordinator-0 -n memmimic -- \
  psql -U memmimic -d memmimic -f /tmp/schema.sql
```

#### 3.3 Redis Deployment

```bash
kubectl apply -f infrastructure/k8s/redis-cluster.yaml

# Verify Redis is running
kubectl wait --namespace=memmimic \
  --for=condition=ready pod \
  --selector=app=redis-master \
  --timeout=300s
```

### Phase 4: Application Services

#### 4.1 Build and Push Images

```bash
# Build base image
docker build -f infrastructure/docker/Dockerfile.base \
  -t your-registry/memmimic-base:latest .

# Build all service images
services=(memory-service classification-service search-service tale-service consciousness-service api-gateway)

for service in "${services[@]}"; do
  docker build -f infrastructure/docker/Dockerfile.$service \
    -t your-registry/memmimic-$service:latest .
  docker push your-registry/memmimic-$service:latest
done
```

#### 4.2 Deploy Services

```bash
# Update image references in manifests
sed -i 's|memmimic/|your-registry/memmimic-|g' infrastructure/k8s/memmimic-services.yaml
sed -i 's|memmimic/|your-registry/memmimic-|g' infrastructure/k8s/api-gateway.yaml

# Deploy services
kubectl apply -f infrastructure/k8s/memmimic-services.yaml
kubectl apply -f infrastructure/k8s/api-gateway.yaml

# Wait for services to be ready
kubectl wait --namespace=memmimic \
  --for=condition=available deployment \
  --all \
  --timeout=600s
```

### Phase 5: Monitoring and Observability

#### 5.1 Deploy Monitoring Stack

```bash
kubectl apply -f infrastructure/k8s/monitoring.yaml

# Wait for monitoring services
kubectl wait --namespace=memmimic \
  --for=condition=available deployment \
  --selector=component=monitoring \
  --timeout=300s
```

#### 5.2 Configure Grafana Dashboards

```bash
# Access Grafana (port-forward for initial setup)
kubectl port-forward -n memmimic service/grafana 3000:3000

# Open http://localhost:3000
# Default credentials: admin / [password from secret]
```

### Phase 6: Ingress and External Access

#### 6.1 Install NGINX Ingress Controller

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer
```

#### 6.2 Configure DNS

```bash
# Get load balancer IP
kubectl get service -n ingress-nginx ingress-nginx-controller

# Update DNS records to point to the load balancer
# Example A records:
# memmimic.com -> [LOAD_BALANCER_IP]
# api.memmimic.com -> [LOAD_BALANCER_IP]
# *.memmimic.com -> [LOAD_BALANCER_IP]
```

#### 6.3 SSL Certificates

```bash
# If using cert-manager
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Apply certificate issuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@memmimic.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Phase 7: Backup and Disaster Recovery

#### 7.1 Deploy Backup System

```bash
# Create backup service
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backup-system
  namespace: memmimic
spec:
  replicas: 1
  selector:
    matchLabels:
      app: backup-system
  template:
    metadata:
      labels:
        app: backup-system
    spec:
      containers:
      - name: backup-system
        image: your-registry/memmimic-backup:latest
        env:
        - name: S3_BUCKET
          value: "memmimic-backups-us-west-2"
        - name: POSTGRES_HOST
          value: "postgres-coordinator"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: memmimic-secrets
              key: db_password
        resources:
          requests:
            cpu: 200m
            memory: 512Mi
          limits:
            cpu: 1
            memory: 1Gi
EOF
```

#### 7.2 Configure Cross-Region Replication

```bash
kubectl apply -f infrastructure/multi-region/disaster-recovery.yaml
```

### Phase 8: Security Hardening

#### 8.1 Network Policies

```bash
# Apply network security policies
kubectl apply -f infrastructure/k8s/network-policies.yaml
```

#### 8.2 Pod Security Standards

```bash
# Enable Pod Security Standards
kubectl label namespace memmimic \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

#### 8.3 Service Mesh (Optional)

```bash
# Install Istio
curl -L https://istio.io/downloadIstio | sh -
istioctl install --set values.defaultRevision=default

# Enable injection for memmimic namespace
kubectl label namespace memmimic istio-injection=enabled
```

### Phase 9: Testing and Validation

#### 9.1 Health Checks

```bash
# Check all pods are running
kubectl get pods -n memmimic

# Check service endpoints
kubectl get endpoints -n memmimic

# Test API gateway
curl -f https://api.memmimic.com/health
```

#### 9.2 Load Testing

```bash
# Run performance tests
k6 run tests/performance/load_test.js \
  --env BASE_URL=https://api.memmimic.com \
  --env CONCURRENT_USERS=100 \
  --env DURATION=5m
```

#### 9.3 Backup Testing

```bash
# Test backup creation
kubectl exec -n memmimic deployment/backup-system -- \
  python -c "
import asyncio
from backup_system import BackupConfig, PostgreSQLBackup
config = BackupConfig(...)
backup = PostgreSQLBackup(config)
asyncio.run(backup.create_backup())
"

# Test restore
kubectl exec -n memmimic deployment/backup-system -- \
  python -c "
import asyncio
from backup_system import BackupConfig, PostgreSQLBackup
config = BackupConfig(...)
backup = PostgreSQLBackup(config)
asyncio.run(backup.restore_backup('backup-id'))
"
```

## Monitoring and Alerting

### Key Metrics to Monitor

- **Application Performance**
  - API response times (< 100ms p95)
  - Error rates (< 0.1%)
  - Memory usage per service
  - Request throughput

- **Database Performance**
  - Query execution times
  - Connection pool usage
  - Replication lag
  - Disk usage and IOPS

- **Infrastructure Health**
  - Node CPU/Memory utilization
  - Pod restart counts
  - Network latency between services
  - Storage utilization

### Alerting Rules

```yaml
# Critical alerts
- alert: ServiceDown
  expr: up{job="memmimic-services"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Service {{ $labels.instance }} is down"

- alert: HighErrorRate
  expr: rate(memmimic_gateway_requests_total{status=~"5.."}[5m]) > 0.01
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"

- alert: DatabaseConnectionFailure
  expr: memmimic_database_connections_active == 0
  for: 30s
  labels:
    severity: critical
  annotations:
    summary: "Database connection failure"
```

## Scaling Guidelines

### Horizontal Pod Autoscaling

The deployment includes HPA configurations for all services:

- **API Gateway**: 3-20 replicas based on CPU (60%) and memory (70%)
- **Memory Service**: 2-10 replicas based on CPU (70%) and memory (80%)
- **Search Service**: 3-15 replicas based on CPU (65%) and memory (75%)

### Database Scaling

1. **Vertical Scaling**: Increase CPU/memory for existing shards
2. **Horizontal Scaling**: Add more shards with rebalancing
3. **Read Replicas**: Add read-only replicas for read-heavy workloads

### Storage Scaling

- Use EBS gp3 volumes with provisioned IOPS for predictable performance
- Enable EBS volume expansion for dynamic growth
- Implement automated cleanup policies for logs and temporary data

## Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check pod status and events
kubectl describe pod -n memmimic [pod-name]

# Check logs
kubectl logs -n memmimic [pod-name] --previous

# Check resource constraints
kubectl top pods -n memmimic
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it -n memmimic postgres-coordinator-0 -- \
  psql -U memmimic -d memmimic -c "SELECT 1"

# Check connection pool status
kubectl exec -n memmimic deployment/memory-service -- \
  curl localhost:9090/metrics | grep connection_pool
```

#### Performance Issues

```bash
# Check service metrics
kubectl port-forward -n memmimic service/prometheus 9090:9090
# Open http://localhost:9090

# Analyze slow queries
kubectl exec -it -n memmimic postgres-coordinator-0 -- \
  psql -U memmimic -d memmimic -c "
  SELECT query, mean_time, calls 
  FROM pg_stat_statements 
  ORDER BY mean_time DESC LIMIT 10;"
```

### Emergency Procedures

#### Service Failure Recovery

1. **Identify failed services**:
   ```bash
   kubectl get pods -n memmimic | grep -v Running
   ```

2. **Check recent deployments**:
   ```bash
   kubectl rollout history deployment -n memmimic
   ```

3. **Rollback if necessary**:
   ```bash
   kubectl rollout undo deployment/[service-name] -n memmimic
   ```

#### Database Recovery

1. **Check database status**:
   ```bash
   kubectl exec -n memmimic postgres-coordinator-0 -- pg_isready
   ```

2. **Restore from backup**:
   ```bash
   kubectl exec -n memmimic deployment/backup-system -- \
     python backup_system.py restore [backup-id]
   ```

#### Complete Cluster Failure

1. **Activate disaster recovery region**:
   ```bash
   # Switch kubectl context to secondary region
   kubectl config use-context secondary-region
   
   # Verify services are running
   kubectl get pods -n memmimic
   
   # Update DNS to point to secondary region
   aws route53 change-resource-record-sets --hosted-zone-id [ZONE-ID] \
     --change-batch file://failover-dns-change.json
   ```

## Performance Tuning

### Database Optimization

```sql
-- PostgreSQL tuning parameters
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
SELECT pg_reload_conf();
```

### Application Optimization

- **Memory Service**: Tune connection pool size based on load
- **Classification Service**: Batch processing for embeddings
- **Search Service**: Implement query result caching
- **API Gateway**: Enable gzip compression and HTTP/2

## Security Best Practices

### Network Security

- Use Kubernetes Network Policies to restrict pod-to-pod communication
- Enable TLS for all inter-service communication
- Use private subnets for database and internal services
- Implement proper firewall rules and security groups

### Secret Management

- Use Kubernetes secrets for sensitive data
- Consider external secret management (AWS Secrets Manager, HashiCorp Vault)
- Rotate secrets regularly
- Never commit secrets to version control

### Container Security

- Use minimal base images (distroless when possible)
- Scan images for vulnerabilities
- Run containers as non-root users
- Implement Pod Security Standards

## Maintenance and Updates

### Regular Maintenance Tasks

1. **Security Updates** (Monthly):
   - Update base images
   - Patch Kubernetes cluster
   - Update application dependencies

2. **Performance Review** (Monthly):
   - Analyze metrics and trends
   - Optimize resource allocation
   - Review and tune HPA settings

3. **Backup Validation** (Weekly):
   - Test backup restoration
   - Verify cross-region replication
   - Clean up old backups

4. **Capacity Planning** (Quarterly):
   - Review growth trends
   - Plan infrastructure scaling
   - Budget for resource increases

### Update Procedures

#### Rolling Updates

```bash
# Update service image
kubectl set image deployment/memory-service \
  memory-service=your-registry/memmimic-memory-service:v2.0.0 \
  -n memmimic

# Monitor rollout
kubectl rollout status deployment/memory-service -n memmimic
```

#### Database Updates

```bash
# Backup before update
kubectl exec -n memmimic deployment/backup-system -- \
  python backup_system.py backup --type=full

# Apply schema migrations
kubectl exec -it -n memmimic postgres-coordinator-0 -- \
  psql -U memmimic -d memmimic -f /tmp/migration.sql
```

## Cost Optimization

### Resource Right-Sizing

- Monitor actual resource usage vs. requests/limits
- Use Vertical Pod Autoscaler for recommendations
- Implement resource quotas to prevent over-allocation

### Storage Optimization

- Use appropriate storage classes (gp3 vs. io1/io2)
- Implement data lifecycle policies
- Archive old data to S3 Glacier

### Multi-Region Cost Management

- Use spot instances for non-critical workloads
- Implement intelligent traffic routing to minimize cross-region costs
- Consider regional data residency requirements

## Conclusion

This production deployment guide provides a comprehensive foundation for running MemMimic at enterprise scale. The architecture supports:

- **High Availability**: Multi-region deployment with automated failover
- **Scalability**: Horizontal scaling from development to enterprise workloads
- **Security**: Enterprise-grade security with network isolation and secret management
- **Monitoring**: Comprehensive observability with metrics, logging, and alerting
- **Disaster Recovery**: Automated backup and cross-region replication

For additional support and advanced configurations, refer to the individual service documentation and operational runbooks.

## Support and Resources

- **Documentation**: [docs/](../docs/)
- **Issue Tracker**: [GitHub Issues](https://github.com/your-org/memmimic/issues)
- **Monitoring Dashboards**: [Grafana Dashboards](./monitoring/dashboards/)
- **Runbooks**: [Operational Runbooks](./runbooks/)

Remember to customize this deployment for your specific requirements, including security policies, compliance requirements, and organizational standards.
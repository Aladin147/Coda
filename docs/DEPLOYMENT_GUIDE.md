# Coda Deployment Guide

> **Production deployment guide for Coda - RTX 5090 optimized voice assistant**

## 🎯 Deployment Overview

This guide covers deploying Coda in production environments with RTX 5090 optimization and comprehensive system integration.

### Deployment Options

1. **RTX 5090 Single Server** - High-performance single-machine deployment
2. **Containerized Deployment** - Docker with NVIDIA GPU support
3. **Cloud Deployment** - AWS/GCP/Azure with GPU instances
4. **Kubernetes Deployment** - Scalable container orchestration
5. **Edge Deployment** - Local/on-premises with RTX 5090

## 🖥️ RTX 5090 Single Server Deployment

### Prerequisites

```bash
# System requirements for RTX 5090 optimization
- Ubuntu 22.04 LTS or Windows 10/11 (22H2+)
- 32GB+ RAM (64GB recommended for optimal performance)
- 500GB+ NVMe SSD storage
- NVIDIA RTX 5090 GPU (24GB VRAM)
- Python 3.11+ (3.11.7 recommended)
- CUDA 12.8+ drivers
- Git 2.40+
```

### Installation Steps

```bash
# 1. Update system and install CUDA 12.8
sudo apt update && sudo apt upgrade -y
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_550.54.15_linux.run
sudo sh cuda_12.8.0_550.54.15_linux.run

# 2. Install Python 3.11 and dependencies
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install build-essential git curl wget portaudio19-dev

# 3. Install RTX 5090 drivers (latest)
sudo apt install nvidia-driver-560
sudo reboot

# 4. Verify RTX 5090 setup
nvidia-smi
python3.11 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 5. Create deployment user
sudo useradd -m -s /bin/bash coda
sudo usermod -aG sudo coda
sudo su - coda

# 6. Clone and setup Coda
git clone https://github.com/Aladin147/Coda.git
cd Coda
python3.11 -m venv venv
source venv/bin/activate

# 7. Install PyTorch nightly with CUDA 12.8 (RTX 5090 support)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 8. Install Coda dependencies
pip install -r requirements.txt

# 9. Set up environment
cp .env.example .env
# Edit .env with your configuration

# 10. Install system services
sudo cp scripts/systemd/coda.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable coda
```

### Production Configuration

```yaml
# configs/production.yaml
system:
  environment: "production"
  debug: false
  log_level: "INFO"
  gpu_optimization: true
  rtx_5090_mode: true

llm:
  provider: "ollama"
  model: "qwen3:30b-a3b"
  base_url: "http://localhost:11434"
  optimization:
    enable_caching: true
    max_concurrent_requests: 8
    gpu_memory_fraction: 0.7
    enable_thinking: false  # Disabled for faster responses

voice:
  enabled: true
  mode: "adaptive"
  provider: "moshi"
  optimization:
    enable_vram_management: true
    max_vram_usage_gb: 12  # RTX 5090 optimized
    enable_streaming: true
    latency_optimization: true

memory:
  provider: "chromadb"
  storage_path: "/var/lib/coda/memory"
  long_term:
    max_memories: 500000  # Increased for production
    embedding_model: "all-MiniLM-L6-v2"
  short_term:
    max_size: 1000

personality:
  enabled: true
  adaptation_rate: 0.1
  memory_integration: true

tools:
  enabled: true
  max_concurrent_tools: 5
  timeout_seconds: 30

dashboard:
  host: "0.0.0.0"
  port: 8080
  security:
    enable_auth: true
    auth_token: "${CODA_AUTH_TOKEN}"
    allowed_origins: ["https://yourdomain.com"]
  features:
    real_time_monitoring: true
    performance_metrics: true

websocket:
  host: "0.0.0.0"
  port: 8765
  security:
    enable_auth: true
    auth_token: "${CODA_WS_TOKEN}"
  optimization:
    max_connections: 100
    heartbeat_interval: 30

performance:
  optimization_level: "rtx_5090"
  targets:
    max_response_time_ms: 500  # RTX 5090 optimized
    max_memory_usage_percent: 75
    gpu_utilization_target: 85
  monitoring:
    enable_metrics: true
    metrics_port: 9090

logging:
  level: "INFO"
  file:
    path: "/var/log/coda/coda.log"
    max_size_mb: 500
    backup_count: 20
  console:
    enabled: true
    format: "structured"
```

### System Service Configuration

```ini
# /etc/systemd/system/coda.service
[Unit]
Description=Coda Assistant Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=coda
Group=coda
WorkingDirectory=/home/coda/coda
Environment=PATH=/home/coda/coda/venv/bin
ExecStart=/home/coda/coda/venv/bin/python coda_launcher.py --config configs/production.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=coda

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/coda /var/log/coda

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/coda
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Dashboard
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API endpoints
    location /api {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🐳 Docker Deployment (RTX 5090 Optimized)

### Dockerfile

```dockerfile
# Dockerfile - RTX 5090 Optimized
FROM nvidia/cuda:12.8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    git \
    curl \
    wget \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash coda
USER coda
WORKDIR /home/coda

# Copy application
COPY --chown=coda:coda . /home/coda/Coda
WORKDIR /home/coda/Coda

# Install Python dependencies
RUN python3.11 -m venv venv
RUN . venv/bin/activate && pip install --upgrade pip

# Install PyTorch nightly with CUDA 12.8 (RTX 5090 support)
RUN . venv/bin/activate && pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install Coda dependencies
RUN . venv/bin/activate && pip install -r requirements.txt

# Create data directories
RUN mkdir -p data/memory data/sessions logs models

# Set environment variables for RTX 5090
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CODA_GPU_OPTIMIZATION=true

# Expose ports
EXPOSE 8080 8765 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start command
CMD ["./venv/bin/python", "-m", "coda.cli", "--config", "configs/production.yaml", "--dashboard"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  coda:
    build: .
    container_name: coda-assistant
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "8765:8765"
    volumes:
      - ./data:/home/coda/coda/data
      - ./logs:/home/coda/coda/logs
      - ./configs:/home/coda/coda/configs
    environment:
      - CODA_AUTH_TOKEN=${CODA_AUTH_TOKEN}
      - CODA_WS_TOKEN=${CODA_WS_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    container_name: coda-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - coda

  ollama:
    image: ollama/ollama:latest
    container_name: coda-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

### Environment Variables

```bash
# .env
CODA_AUTH_TOKEN=your-secure-auth-token-here
CODA_WS_TOKEN=your-secure-websocket-token-here
CODA_LOG_LEVEL=INFO
CODA_ENVIRONMENT=production
```

## ☁️ Cloud Deployment

### AWS Deployment

#### EC2 Instance Setup

```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type g4dn.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --user-data file://user-data.sh

# user-data.sh
#!/bin/bash
yum update -y
yum install -y docker git
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Clone and deploy Coda
git clone https://github.com/your-repo/coda.git /home/ec2-user/coda
cd /home/ec2-user/coda
docker-compose up -d
```

#### Application Load Balancer

```yaml
# cloudformation-template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  CodaLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: coda-alb
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref ALBSecurityGroup

  CodaTargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: coda-targets
      Port: 8080
      Protocol: HTTP
      VpcId: !Ref VPC
      HealthCheckPath: /api/health
      HealthCheckProtocol: HTTP
      HealthCheckIntervalSeconds: 30
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3

  CodaListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref CodaTargetGroup
      LoadBalancerArn: !Ref CodaLoadBalancer
      Port: 443
      Protocol: HTTPS
      Certificates:
        - CertificateArn: !Ref SSLCertificate
```

### Google Cloud Platform

```yaml
# gcp-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coda-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: coda
  template:
    metadata:
      labels:
        app: coda
    spec:
      containers:
      - name: coda
        image: gcr.io/your-project/coda:latest
        ports:
        - containerPort: 8080
        - containerPort: 8765
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        env:
        - name: CODA_AUTH_TOKEN
          valueFrom:
            secretKeyRef:
              name: coda-secrets
              key: auth-token
---
apiVersion: v1
kind: Service
metadata:
  name: coda-service
spec:
  selector:
    app: coda
  ports:
  - name: dashboard
    port: 80
    targetPort: 8080
  - name: websocket
    port: 8765
    targetPort: 8765
  type: LoadBalancer
```

## 🔧 Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: coda-system

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: coda-config
  namespace: coda-system
data:
  production.yaml: |
    system:
      environment: "production"
      log_level: "INFO"
    llm:
      provider: "ollama"
      model: "qwen3:30b-a3b"
    # ... rest of config

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: coda-secrets
  namespace: coda-system
type: Opaque
data:
  auth-token: <base64-encoded-token>
  ws-token: <base64-encoded-token>

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coda-deployment
  namespace: coda-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: coda
  template:
    metadata:
      labels:
        app: coda
    spec:
      containers:
      - name: coda
        image: your-registry/coda:latest
        ports:
        - containerPort: 8080
          name: dashboard
        - containerPort: 8765
          name: websocket
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        env:
        - name: CODA_AUTH_TOKEN
          valueFrom:
            secretKeyRef:
              name: coda-secrets
              key: auth-token
        - name: CODA_WS_TOKEN
          valueFrom:
            secretKeyRef:
              name: coda-secrets
              key: ws-token
        volumeMounts:
        - name: config-volume
          mountPath: /home/coda/coda/configs
        - name: data-volume
          mountPath: /home/coda/coda/data
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config-volume
        configMap:
          name: coda-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: coda-data-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: coda-service
  namespace: coda-system
spec:
  selector:
    app: coda
  ports:
  - name: dashboard
    port: 80
    targetPort: 8080
  - name: websocket
    port: 8765
    targetPort: 8765
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: coda-ingress
  namespace: coda-system
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/websocket-services: "coda-service"
spec:
  tls:
  - hosts:
    - yourdomain.com
    secretName: coda-tls
  rules:
  - host: yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: coda-service
            port:
              number: 80
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: coda-service
            port:
              number: 8765
```

### Deployment Commands

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n coda-system
kubectl get services -n coda-system
kubectl get ingress -n coda-system

# View logs
kubectl logs -f deployment/coda-deployment -n coda-system

# Scale deployment
kubectl scale deployment coda-deployment --replicas=3 -n coda-system
```

## 📊 Monitoring & Observability

### Prometheus Monitoring

```yaml
# monitoring/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'coda'
      static_configs:
      - targets: ['coda-service:9090']
      metrics_path: /metrics
      scrape_interval: 30s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Coda System Metrics",
    "panels": [
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "coda_response_time_seconds",
            "legendFormat": "Response Time"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "coda_memory_usage_percent",
            "legendFormat": "Memory %"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Considerations

### Security Checklist

- [ ] Enable HTTPS/TLS encryption
- [ ] Configure authentication tokens
- [ ] Set up firewall rules
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] Network segmentation
- [ ] Access control policies

### Backup Strategy

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/coda_$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup data
tar -czf $BACKUP_DIR/data.tar.gz data/
tar -czf $BACKUP_DIR/configs.tar.gz configs/
tar -czf $BACKUP_DIR/logs.tar.gz logs/

# Upload to cloud storage
aws s3 cp $BACKUP_DIR s3://your-backup-bucket/coda/ --recursive

# Cleanup old backups (keep last 30 days)
find /backups -name "coda_*" -mtime +30 -exec rm -rf {} \;
```

---

**🚀 Your Coda deployment is now ready for production! Monitor performance and scale as needed.**

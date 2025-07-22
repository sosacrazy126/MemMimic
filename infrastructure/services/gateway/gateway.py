#!/usr/bin/env python3
"""
MemMimic Production API Gateway
Service orchestration, load balancing, and request routing
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('memmimic_gateway_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('memmimic_gateway_request_duration_seconds', 'Request duration', ['service'])
SERVICE_HEALTH = Gauge('memmimic_service_health', 'Service health status', ['service'])
ACTIVE_CONNECTIONS = Gauge('memmimic_gateway_active_connections', 'Active connections')


class ServiceConfig:
    """Service configuration and health tracking"""
    def __init__(self, name: str, url: str, timeout: float = 30.0):
        self.name = name
        self.url = url
        self.timeout = timeout
        self.is_healthy = True
        self.last_check = datetime.now()
        self.consecutive_failures = 0
        self.response_times: List[float] = []


class LoadBalancer:
    """Round-robin load balancer with health checks"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceConfig]] = {}
        self.current_index: Dict[str, int] = {}
        self.health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
    def register_service(self, service_type: str, services: List[ServiceConfig]):
        """Register service instances for load balancing"""
        self.services[service_type] = services
        self.current_index[service_type] = 0
        logger.info(f"Registered {len(services)} instances for {service_type}")
        
    def get_service(self, service_type: str) -> Optional[ServiceConfig]:
        """Get next healthy service instance using round-robin"""
        if service_type not in self.services:
            return None
            
        instances = self.services[service_type]
        healthy_instances = [s for s in instances if s.is_healthy]
        
        if not healthy_instances:
            logger.error(f"No healthy instances for {service_type}")
            return None
            
        # Round-robin through healthy instances
        current_idx = self.current_index[service_type]
        selected = healthy_instances[current_idx % len(healthy_instances)]
        self.current_index[service_type] = (current_idx + 1) % len(healthy_instances)
        
        return selected
        
    async def start_health_checks(self):
        """Start periodic health checks"""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
    async def stop_health_checks(self):
        """Stop health checks"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
    async def _health_check_loop(self):
        """Periodic health check loop"""
        while True:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
                
    async def _check_all_services(self):
        """Check health of all registered services"""
        for service_type, instances in self.services.items():
            for service in instances:
                await self._check_service_health(service)
                
    async def _check_service_health(self, service: ServiceConfig):
        """Check health of individual service"""
        try:
            start_time = time.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service.url}/health")
                
            response_time = time.time() - start_time
            service.response_times.append(response_time)
            service.response_times = service.response_times[-10:]  # Keep last 10
            
            if response.status_code == 200:
                if not service.is_healthy:
                    logger.info(f"Service {service.name} is now healthy")
                service.is_healthy = True
                service.consecutive_failures = 0
                SERVICE_HEALTH.labels(service=service.name).set(1)
            else:
                service.consecutive_failures += 1
                if service.consecutive_failures >= 3 and service.is_healthy:
                    logger.error(f"Service {service.name} became unhealthy")
                    service.is_healthy = False
                    SERVICE_HEALTH.labels(service=service.name).set(0)
                    
        except Exception as e:
            service.consecutive_failures += 1
            if service.consecutive_failures >= 3 and service.is_healthy:
                logger.error(f"Service {service.name} became unhealthy: {e}")
                service.is_healthy = False
                SERVICE_HEALTH.labels(service=service.name).set(0)
                
        service.last_check = datetime.now()


class APIGateway:
    """Main API Gateway class"""
    
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.http_client: Optional[httpx.AsyncClient] = None
        self._setup_services()
        
    def _setup_services(self):
        """Configure service instances"""
        # In production, these would come from service discovery
        services_config = {
            "memory": [
                ServiceConfig("memory-service-1", "http://memory-service:8001"),
            ],
            "classification": [
                ServiceConfig("classification-service-1", "http://classification-service:8002", timeout=60.0),
            ],
            "search": [
                ServiceConfig("search-service-1", "http://search-service:8003"),
            ],
            "tale": [
                ServiceConfig("tale-service-1", "http://tale-service:8004"),
            ],
            "consciousness": [
                ServiceConfig("consciousness-service-1", "http://consciousness-service:8005"),
            ]
        }
        
        for service_type, instances in services_config.items():
            self.load_balancer.register_service(service_type, instances)
            
    async def initialize(self):
        """Initialize gateway"""
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        await self.load_balancer.start_health_checks()
        logger.info("API Gateway initialized")
        
    async def shutdown(self):
        """Shutdown gateway"""
        await self.load_balancer.stop_health_checks()
        if self.http_client:
            await self.http_client.aclose()
        logger.info("API Gateway shutdown complete")
        
    async def route_request(self, service_type: str, path: str, method: str, 
                          body: Optional[bytes] = None, 
                          headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, Any], str]:
        """Route request to appropriate service"""
        service = self.load_balancer.get_service(service_type)
        if not service:
            raise HTTPException(status_code=503, detail=f"No healthy {service_type} service available")
            
        url = f"{service.url}{path}"
        request_headers = headers or {}
        
        # Remove hop-by-hop headers
        hop_by_hop_headers = {
            'connection', 'keep-alive', 'proxy-authenticate', 
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
        }
        request_headers = {k: v for k, v in request_headers.items() 
                          if k.lower() not in hop_by_hop_headers}
        
        start_time = time.time()
        try:
            if method.upper() == "GET":
                response = await self.http_client.get(url, headers=request_headers, timeout=service.timeout)
            elif method.upper() == "POST":
                response = await self.http_client.post(url, content=body, headers=request_headers, timeout=service.timeout)
            elif method.upper() == "PUT":
                response = await self.http_client.put(url, content=body, headers=request_headers, timeout=service.timeout)
            elif method.upper() == "DELETE":
                response = await self.http_client.delete(url, headers=request_headers, timeout=service.timeout)
            else:
                response = await self.http_client.request(method, url, content=body, headers=request_headers, timeout=service.timeout)
                
            duration = time.time() - start_time
            REQUEST_DURATION.labels(service=service_type).observe(duration)
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = {"content": response.text}
                
            return response.status_code, response_data, response.headers.get("content-type", "application/json")
            
        except httpx.TimeoutException:
            logger.error(f"Timeout calling {service_type} service at {url}")
            raise HTTPException(status_code=504, detail=f"{service_type} service timeout")
        except Exception as e:
            logger.error(f"Error calling {service_type} service: {e}")
            raise HTTPException(status_code=502, detail=f"{service_type} service error")


# Global gateway instance
gateway = APIGateway()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    await gateway.initialize()
    yield
    await gateway.shutdown()

# Create FastAPI app
app = FastAPI(
    title="MemMimic API Gateway",
    description="Production API Gateway for MemMimic Memory System",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics for all requests"""
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
    finally:
        ACTIVE_CONNECTIONS.dec()


@app.get("/health")
async def health_check():
    """Gateway health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/status")
async def gateway_status():
    """Detailed gateway status"""
    service_status = {}
    
    for service_type, instances in gateway.load_balancer.services.items():
        service_status[service_type] = {
            "total_instances": len(instances),
            "healthy_instances": sum(1 for s in instances if s.is_healthy),
            "instances": [
                {
                    "name": s.name,
                    "url": s.url,
                    "healthy": s.is_healthy,
                    "last_check": s.last_check.isoformat(),
                    "consecutive_failures": s.consecutive_failures,
                    "avg_response_time": sum(s.response_times) / len(s.response_times) if s.response_times else 0
                }
                for s in instances
            ]
        }
    
    return {
        "gateway_status": "healthy",
        "services": service_status,
        "timestamp": datetime.now().isoformat()
    }


# Service routing endpoints
@app.api_route("/api/memory/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_memory_service(request: Request, path: str):
    """Route requests to memory service"""
    body = await request.body() if request.method in ["POST", "PUT"] else None
    headers = dict(request.headers)
    
    status_code, data, content_type = await gateway.route_request(
        "memory", f"/{path}", request.method, body, headers
    )
    
    return JSONResponse(content=data, status_code=status_code)


@app.api_route("/api/classification/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_classification_service(request: Request, path: str):
    """Route requests to classification service"""
    body = await request.body() if request.method in ["POST", "PUT"] else None
    headers = dict(request.headers)
    
    status_code, data, content_type = await gateway.route_request(
        "classification", f"/{path}", request.method, body, headers
    )
    
    return JSONResponse(content=data, status_code=status_code)


@app.api_route("/api/search/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_search_service(request: Request, path: str):
    """Route requests to search service"""
    body = await request.body() if request.method in ["POST", "PUT"] else None
    headers = dict(request.headers)
    
    status_code, data, content_type = await gateway.route_request(
        "search", f"/{path}", request.method, body, headers
    )
    
    return JSONResponse(content=data, status_code=status_code)


@app.api_route("/api/tales/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_tale_service(request: Request, path: str):
    """Route requests to tale service"""
    body = await request.body() if request.method in ["POST", "PUT"] else None
    headers = dict(request.headers)
    
    status_code, data, content_type = await gateway.route_request(
        "tale", f"/{path}", request.method, body, headers
    )
    
    return JSONResponse(content=data, status_code=status_code)


@app.api_route("/api/consciousness/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_consciousness_service(request: Request, path: str):
    """Route requests to consciousness service"""
    body = await request.body() if request.method in ["POST", "PUT"] else None
    headers = dict(request.headers)
    
    status_code, data, content_type = await gateway.route_request(
        "consciousness", f"/{path}", request.method, body, headers
    )
    
    return JSONResponse(content=data, status_code=status_code)


# MCP Tool integration endpoints
@app.post("/mcp/recall_cxd")
async def mcp_recall_cxd(request: Request):
    """MCP recall_cxd tool endpoint"""
    body = await request.body()
    headers = dict(request.headers)
    
    status_code, data, content_type = await gateway.route_request(
        "search", "/recall_cxd", "POST", body, headers
    )
    
    return JSONResponse(content=data, status_code=status_code)


@app.post("/mcp/remember")
async def mcp_remember(request: Request):
    """MCP remember tool endpoint"""
    body = await request.body()
    headers = dict(request.headers)
    
    status_code, data, content_type = await gateway.route_request(
        "memory", "/remember", "POST", body, headers
    )
    
    return JSONResponse(content=data, status_code=status_code)


@app.post("/mcp/remember_with_quality")
async def mcp_remember_with_quality(request: Request):
    """MCP remember_with_quality tool endpoint"""
    body = await request.body()
    headers = dict(request.headers)
    
    status_code, data, content_type = await gateway.route_request(
        "memory", "/remember_with_quality", "POST", body, headers
    )
    
    return JSONResponse(content=data, status_code=status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_config=None,
        access_log=False
    )
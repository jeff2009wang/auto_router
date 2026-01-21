"""
中间件模块
包含请求处理、性能监控等中间件
"""
import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import psutil
import torch


class PerformanceMiddleware(BaseHTTPMiddleware):
    """性能监控中间件"""
    
    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # 记录请求信息
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)
        
        self.logger.info(f"📨 {method} {url} from {client_ip}")
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 添加响应头
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Response-Time"] = f"{process_time*1000:.2f}ms"
            
            # 记录性能日志
            if process_time > 1.0:  # 超过1秒的请求
                self.logger.warning(f"⚠️ Slow request: {method} {url} took {process_time*1000:.2f}ms")
            else:
                self.logger.info(f"✅ {method} {url} completed in {process_time*1000:.2f}ms")
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            self.logger.error(f"❌ {method} {url} failed after {process_time*1000:.2f}ms: {str(e)}")
            raise


class ResourceMonitoringMiddleware(BaseHTTPMiddleware):
    """资源监控中间件"""
    
    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger
        self.last_check = time.time()
        self.check_interval = 60  # 每60秒检查一次
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        current_time = time.time()
        
        # 定期检查系统资源
        if current_time - self.last_check > self.check_interval:
            self._check_resources()
            self.last_check = current_time
        
        return await call_next(request)
    
    def _check_resources(self):
        """检查系统资源使用情况"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            
            # GPU内存使用率（如果可用）
            gpu_memory_info = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    total = torch.cuda.get_device_properties(i).total_memory
                    reserved = torch.cuda.memory_reserved(i)
                    allocated = torch.cuda.memory_allocated(i)
                    free = total - allocated
                    
                    gpu_memory_info[f"gpu_{i}"] = {
                        "total": total,
                        "allocated": allocated,
                        "reserved": reserved,
                        "free": free,
                        "usage_percent": (allocated / total) * 100
                    }
            
            # 记录资源使用情况
            if cpu_percent > 80:
                self.logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 85:
                self.logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
            
            for gpu_id, info in gpu_memory_info.items():
                if info["usage_percent"] > 90:
                    self.logger.warning(f"⚠️ High {gpu_id} memory usage: {info['usage_percent']:.1f}%")
            
            self.logger.debug(
                f"📊 Resources - CPU: {cpu_percent:.1f}%, "
                f"Memory: {memory.percent:.1f}%, "
                f"GPU: {len(gpu_memory_info)} devices"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check resources: {e}")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """错误处理中间件"""
    
    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            self.logger.error(f"Unhandled error in {request.method} {request.url}: {str(e)}")
            
            # 返回友好的错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "timestamp": time.time()
                }
            )


class SecurityMiddleware(BaseHTTPMiddleware):
    """安全中间件"""
    
    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger
        self.request_count = {}
        self.rate_limit = 100  # 每分钟最大请求数
        self.time_window = 60  # 时间窗口（秒）
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # 清理过期的请求记录
        self._cleanup_request_history(client_ip, current_time)
        
        # 检查请求频率
        if not self._check_rate_limit(client_ip, current_time):
            self.logger.warning(f"🚫 Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        response = await call_next(request)
        
        # 添加安全头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response
    
    def _cleanup_request_history(self, client_ip: str, current_time: float):
        """清理过期的请求历史"""
        if client_ip in self.request_count:
            # 移除超过时间窗口的请求
            self.request_count[client_ip] = [
                timestamp for timestamp in self.request_count[client_ip]
                if current_time - timestamp < self.time_window
            ]
    
    def _check_rate_limit(self, client_ip: str, current_time: float) -> bool:
        """检查请求频率限制"""
        if client_ip not in self.request_count:
            self.request_count[client_ip] = []
        
        # 添加当前请求时间戳
        self.request_count[client_ip].append(current_time)
        
        # 检查是否超过限制
        return len(self.request_count[client_ip]) <= self.rate_limit


def setup_middleware(app, logger: logging.Logger):
    """设置所有中间件"""
    # 添加中间件（顺序很重要）
    app.add_middleware(PerformanceMiddleware, logger=logger)
    app.add_middleware(ResourceMonitoringMiddleware, logger=logger)
    app.add_middleware(SecurityMiddleware, logger=logger)
    app.add_middleware(ErrorHandlingMiddleware, logger=logger)
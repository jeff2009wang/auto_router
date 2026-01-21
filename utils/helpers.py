"""
工具函数模块
包含通用的辅助函数和工具
"""
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from functools import wraps
from datetime import datetime


def setup_logging(name: str = "NekoBrain", level: str = "INFO") -> logging.Logger:
    """设置日志配置"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def timing_decorator(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} took {(end_time - start_time)*1000:.2f}ms")
        
        return result
    return wrapper


def get_text_hash(text: str) -> str:
    """生成文本的MD5哈希值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """归一化分数到1-10范围"""
    if not raw_scores:
        return raw_scores
    
    scores = list(raw_scores.values())
    min_score, max_score = min(scores), max(scores)
    
    if max_score == min_score:
        return {label: 5.0 for label in raw_scores.keys()}
    
    return {
        label: 1.0 + 9.0 * (score - min_score) / (max_score - min_score)
        for label, score in raw_scores.items()
    }


def extract_text_from_messages(messages: List[Dict]) -> str:
    """从消息列表中提取纯文本"""
    text_parts = []
    
    for message in messages:
        content = message.get("content", "")
        
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
    
    return " ".join(text_parts)


def has_image_content(messages: List[Dict]) -> bool:
    """检查消息是否包含图像"""
    for message in messages[-2:]:  # 检查最近两条消息
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if item.get("type") in ["image", "image_url"]:
                    return True
    return False


def format_router_response(label: str, scores: Dict[str, float], processing_time: float) -> str:
    """格式化路由响应信息"""
    lines = [
        "=" * 60,
        f"🎯 Final Decision: {label}",
        f"⏱️ Processing Time: {processing_time*1000:.1f}ms",
        "-" * 60,
        "📊 Scoring Details:"
    ]
    
    for category, score in scores.items():
        lines.append(f"  {category:15} | Score: {score:.2f}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def validate_messages(messages: List[Dict]) -> bool:
    """验证消息格式"""
    if not isinstance(messages, list):
        return False
    
    for message in messages:
        if not isinstance(message, dict):
            return False
        
        if "role" not in message or "content" not in message:
            return False
        
        role = message["role"]
        if role not in ["system", "user", "assistant"]:
            return False
    
    return True


def get_model_info() -> Dict[str, Any]:
    """获取模型信息"""
    import torch
    from config.settings import settings
    
    return {
        "model_id": settings.UNIFIED_MODEL_ID,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_gpu": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "gpu_memory": {
            f"gpu_{i}": torch.cuda.get_device_properties(i).total_memory 
            for i in range(torch.cuda.device_count())
        } if torch.cuda.is_available() else {}
    }


def clean_cache_entry(cache_dict: Dict, max_age_hours: int = 24) -> None:
    """清理过期的缓存条目"""
    current_time = time.time()
    expired_keys = []
    
    for key, (value, timestamp) in cache_dict.items():
        if current_time - timestamp > max_age_hours * 3600:
            expired_keys.append(key)
    
    for key in expired_keys:
        del cache_dict[key]


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    import psutil
    import torch
    
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_total": format_file_size(psutil.virtual_memory().total),
        "memory_available": format_file_size(psutil.virtual_memory().available),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": {
            "total": format_file_size(psutil.disk_usage('/').total),
            "used": format_file_size(psutil.disk_usage('/').used),
            "free": format_file_size(psutil.disk_usage('/').free),
            "percent": psutil.disk_usage('/').percent
        },
        "gpu_info": get_model_info()
    }
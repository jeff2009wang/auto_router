"""
项目配置文件
针对RTX 2060 12GB优化的设置
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """应用配置"""
    
    # 应用基础配置
    APP_NAME: str = "Unified NekoBrain"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default_factory=lambda: os.getenv("NEKOBRAIN_DEBUG", "false").lower() == "true", description="调试模式")
    HOST: str = "0.0.0.0"
    PORT: int = 2001
    
    # GPU配置 - 针对RTX 2060 12GB优化
    CUDA_VISIBLE_DEVICES: str = "0"
    PYTORCH_CUDA_ALLOC_CONF: str = "expandable_segments:True"
    
    # 模型配置 - 使用Qwen2.5-VL-3B，更适合RTX 2060 12GB
    UNIFIED_MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    MODEL_CACHE_DIR: str = "./models"
    
    # 量化配置 - 更激进的4bit量化适配12GB显存
    LOAD_IN_4BIT: bool = True
    BNB_4BIT_USE_DOUBLE_QUANT: bool = True
    BNB_4BIT_QUANT_TYPE: str = "nf4"
    BNB_4BIT_COMPUTE_DTYPE: str = "float16"
    MAX_MEMORY: Dict[str, str] = Field(default_factory=lambda: {0: "6GB", "cpu": "16GB"})
    
    # 内存优化配置
    LOW_CPU_MEM_USAGE: bool = True
    OFFLOAD_FOLDER: str = "./offload"
    OFFLOAD_STATE_DICT: bool = True
    
    # 性能优化配置 - 针对RTX 2060 12GB优化
    MAX_WORKERS: int = 1  # 减少工作线程以节省内存
    MAX_NEW_TOKENS: int = 50  # 减少生成token数量
    ENABLE_TORCH_COMPILE: bool = False  # 禁用torch.compile以节省内存
    ENABLE_KV_CACHE: bool = True
    
    # 内存管理配置
    CLEAR_CACHE_AFTER_INFERENCE: bool = True
    MEMORY_CLEANUP_INTERVAL: int = 10  # 每10次推理后清理内存
    
    # 缓存配置
    ROUTE_CACHE_SIZE: int = 256
    ROUTE_CACHE_TTL: int = 3600
    
    # 聚合API配置
    AGGREGATOR_API_KEY: str = Field(default_factory=lambda: os.getenv("AGGREGATOR_API_KEY", ""))
    AGGREGATOR_BASE_URL: str = Field(default_factory=lambda: os.getenv("AGGREGATOR_BASE_URL", "http://192.168.50.165:3000/v1"))
    
    # 目标模型映射
    MODEL_MAP: Dict[str, str] = Field(default_factory=lambda: {
        "flash_smart": "gemini-3-flash-preview",
        "pro_advanced": "gemini-3-pro-preview", 
        "code_technical": "gpt-5-codex-high",
        "code_architect": "claude-4-opus",
        "logic_reasoning": "gemini-3-pro-deepthink",
        "expert_xhigh": "gpt-5.2-xhigh"
    })
    
    # 路由类别定义
    ROUTING_CATEGORIES: Dict[str, str] = Field(default_factory=lambda: {
        "flash_smart": "General chat, greetings, simple questions, daily conversation",
        "pro_advanced": "Complex analysis, creative writing, nuanced language understanding, detailed explanations",
        "code_technical": "Programming, debugging, SQL queries, writing code in Python/C++/Java, technical scripts",
        "code_architect": "System design, software architecture, explaining technical concepts, architectural patterns",
        "logic_reasoning": "Math proofs, physics problems, logic puzzles, step-by-step reasoning, calculus, theorems",
        "expert_xhigh": "Professional research, academic papers, high-context analysis, specialized knowledge"
    })
    
    # 快速路径关键词
    QUICK_KEYWORDS: Dict[str, List[str]] = Field(default_factory=lambda: {
        "code_technical": ["def ", "class ", "import ", "function", "sql", "query", "python", "javascript", "java", "c++", "代码", "编程", "debug"],
        "code_architect": ["architecture", "design pattern", "system design", "microservice", "架构", "设计模式"],
        "logic_reasoning": ["prove", "theorem", "calculate", "solve", "equation", "integral", "微分", "积分", "证明", "计算"],
        "pro_advanced": ["creative", "story", "poem", "creative writing", "创作", "故事", "诗歌", "analysis"],
        "flash_smart": ["hello", "hi", "thanks", "你好", "谢谢"],
        "expert_xhigh": ["research", "paper", "academic", "research", "研究", "学术"]
    })
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    LOG_FILE: str = "./logs/neko_brain.log"
    
    # CORS配置
    CORS_ALLOW_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_METHODS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default_factory=lambda: ["*"])
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()
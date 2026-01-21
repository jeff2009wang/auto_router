"""
统一NekoBrain API服务
整合VLM和决策模型为单一服务
"""
import os
import logging
import json
import traceback
import asyncio
import time
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import litellm

from src.model_manager import UnifiedModelManager
from config.settings import settings


# 设置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(settings.APP_NAME)

# 抑制第三方库日志
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
litellm.suppress_debug_info = True


# 全局模型管理器实例
brain: UnifiedModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global brain
    
    logger.info("🚀 Starting Unified NekoBrain Service...")
    
    try:
        # 初始化模型管理器
        brain = UnifiedModelManager()
        brain._setup_logging()
        brain.load_model()
        
        logger.info("✅ Service initialized successfully")
        logger.info(f"🎯 Model: {settings.UNIFIED_MODEL_ID}")
        logger.info(f"💾 Device: {brain.device}")
        logger.info(f"🔧 Workers: {settings.MAX_WORKERS}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        logger.error(traceback.format_exc())
        raise e
    
    yield
    
    # 清理资源
    logger.info("🛑 Shutting down service...")
    if brain:
        brain.route_cache.clear()
    logger.info("✅ Service shutdown complete")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="统一VLM和决策模型服务 - 针对RTX 2060 12GB优化",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# 请求模型
class ChatRequest(BaseModel):
    messages: List[Dict]
    model: str
    stream: Optional[bool] = True


# 响应模型
class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    if brain is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return HealthResponse(
        status="healthy",
        model=settings.UNIFIED_MODEL_ID,
        device=brain.device,
        version=settings.APP_VERSION
    )


@app.get("/models")
async def list_models():
    """列出可用的路由模型"""
    return {
        "unified_model": settings.UNIFIED_MODEL_ID,
        "target_models": settings.MODEL_MAP,
        "routing_categories": settings.ROUTING_CATEGORIES
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """统一的聊天完成接口"""
    try:
        if brain is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # 统一处理auto_router模型名称
        if req.model and req.model.startswith("auto"):
            req.model = "auto_router"
        
        logger.info(f"📨 Received request: {req.model}")
        
        # 记录决策开始时间
        decision_start = time.time()
        
        # 使用统一模型进行路由决策
        label, processed_msgs = await brain.route(req.messages)
        
        # 获取目标模型
        target_model = settings.MODEL_MAP.get(label, "gemini-3-flash-preview")
        
        # 为代码和逻辑任务注入助手提示
        if "code" in label or "logic" in label:
            processed_msgs = brain.inject_assistant_prompt(processed_msgs)
        
        logger.info(f"🎯 Routing to: {target_model} (category: {label})")
        
        # 调用聚合API
        resp = await litellm.acompletion(
            model=f"openai/{target_model}",
            messages=processed_msgs,
            stream=req.stream,
            api_base=settings.AGGREGATOR_BASE_URL,
            api_key=settings.AGGREGATOR_API_KEY
        )
        
        # 处理流式响应
        if req.stream:
            async def generate_stream():
                # 发送路由信息前缀 - 更美观的显示
                category_names = {
                    'flash_smart': '💬 通用对话',
                    'pro_advanced': '📝 高级分析',
                    'code_technical': '💻 技术编程',
                    'code_architect': '🏗️ 架构设计',
                    'logic_reasoning': '🧮 逻辑推理',
                    'expert_xhigh': '🎓 专业研究'
                }
                
                category_name = category_names.get(label, label)
                model_names = {
                    'gemini-3-flash-preview': 'Gemini 3 Flash',
                    'gemini-3-pro-preview': 'Gemini 3 Pro',
                    'gpt-5-codex-high': 'GPT-5 Codex',
                    'claude-4-opus': 'Claude 4 Opus',
                    'gemini-3-pro-deepthink': 'Gemini 3 DeepThink',
                    'gpt-5.2-xhigh': 'GPT-5.2 XHigh'
                }
                
                model_name = model_names.get(target_model, target_model)
                
                prefix = f"> 🧠 **Unified NekoBrain v2.0**\n> 🎯 智能路由: {category_name}\n> 🤖 目标模型: {model_name}\n> ⚡ 推理时间: ~{((time.time() - decision_start)*1000):.0f}ms\n\n"
                yield f"data: {json.dumps({'choices': [{'delta': {'content': prefix}, 'index': 0}], 'model': target_model})}\n\n"
                
                # 流式发送响应
                async for chunk in resp:
                    yield f"data: {chunk.model_dump_json()}\n\n"
                
                # 结束标记
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        # 非流式响应
        return resp
        
    except Exception as e:
        logger.error(f"❌ Request failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """获取服务统计信息"""
    if brain is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "cache_size": len(brain.route_cache.cache),
        "cache_max_size": brain.route_cache.max_size,
        "device": brain.device,
        "model": settings.UNIFIED_MODEL_ID,
        "performance_logging": brain.enable_perf_logging
    }


@app.post("/clear_cache")
async def clear_cache():
    """清空路由缓存"""
    if brain is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    brain.route_cache.clear()
    logger.info("🧹 Route cache cleared")
    
    return {"status": "cache_cleared", "message": "路由缓存已清空"}


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model": settings.UNIFIED_MODEL_ID,
        "description": "统一VLM和决策模型服务",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "chat": "/v1/chat/completions",
            "stats": "/stats",
            "clear_cache": "/clear_cache"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"🌐 Server: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"🎯 Model: {settings.UNIFIED_MODEL_ID}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,  # 关闭reload模式以避免日志干扰
        log_level=settings.LOG_LEVEL.lower()
    )
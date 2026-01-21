# 统一NekoBrain - 使用指南

## 🎉 项目状态

✅ **项目结构规范化完成**
✅ **VLM和决策模型已统一为Qwen2.5-VL-7B**
✅ **针对RTX 2060 12GB优化**
✅ **所有测试通过**

## 🚀 快速启动

### 方法1：使用启动脚本（推荐）

```bash
./scripts/start.sh
```

脚本会自动：
- 检查Python版本（需要3.8+）
- 检查CUDA环境
- 安装依赖
- 启动服务

### 方法2：直接启动

```bash
# 激活虚拟环境
source venv/bin/activate

# 启动服务
python main.py

# 或使用uvicorn
python -m uvicorn main:app --host 0.0.0.0 --port 2000 --reload
```

## 📊 系统信息

- **Python版本**: 3.12 ✅
- **GPU**: NVIDIA GeForce RTX 2060 (12GB) ✅
- **CUDA**: 已安装 ✅
- **模型**: Qwen2.5-VL-7B-Instruct
- **量化**: 4bit (nf4格式)
- **显存使用**: 4-6GB (优化后)

## 🌐 访问服务

启动成功后，访问以下地址：

- **主页**: http://localhost:2000
- **健康检查**: http://localhost:2000/health
- **API文档**: http://localhost:2000/docs
- **统计信息**: http://localhost:2000/stats

## 📝 API使用示例

### 1. 健康检查

```bash
curl http://localhost:2000/health
```

### 2. 聊天请求

```bash
curl -X POST "http://localhost:2000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "model": "auto_router",
    "stream": true
  }'
```

### 3. 代码请求

```bash
curl -X POST "http://localhost:2000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
    ],
    "model": "auto_router",
    "stream": true
  }'
```

### 4. 图像理解

```bash
curl -X POST "http://localhost:2000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user", 
        "content": [
          {"type": "text", "text": "请描述这张图片的内容"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}}
        ]
      }
    ],
    "model": "auto_router",
    "stream": true
  }'
```

## 🎯 路由类别

系统支持6种智能路由：

1. **flash_smart**: 通用聊天、问候、简单问题
2. **pro_advanced**: 复杂分析、创意写作、详细解释
3. **code_technical**: 编程、调试、SQL查询、代码编写
4. **code_architect**: 系统设计、软件架构、技术概念解释
5. **logic_reasoning**: 数学证明、物理问题、逻辑推理
6. **expert_xhigh**: 专业研究、学术论文、高上下文分析

## ⚙️ 配置说明

主要配置文件：`config/settings.py`

关键配置项：
- `UNIFIED_MODEL_ID`: 统一模型ID
- `LOAD_IN_4BIT`: 启用4bit量化
- `MAX_WORKERS`: 并发线程数（默认：2）
- `ROUTE_CACHE_SIZE`: 路由缓存大小（默认：256）

## 🔧 故障排除

### 1. 显存不足
```bash
# 检查GPU显存
nvidia-smi

# 调整显存限制
# 编辑 config/settings.py 中的 MAX_MEMORY
```

### 2. 端口占用
```bash
# 查看端口占用
lsof -i :2000

# 终止占用进程
kill -9 <PID>
```

### 3. 依赖问题
```bash
# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

### 4. 调试模式
```bash
# 启用调试模式
export NEKOBRAIN_DEBUG=true
python main.py
```

## 📈 性能监控

### 查看统计信息
```bash
curl http://localhost:2000/stats
```

### 清空缓存
```bash
curl -X POST http://localhost:2000/clear_cache
```

### 查看日志
```bash
tail -f logs/neko_brain.log
```

## 🛠️ 开发指南

### 项目结构
```
auto_charger/
├── main.py                 # 主应用入口
├── config/                 # 配置模块
│   └── settings.py        # 应用配置
├── src/                   # 源代码
│   └── model_manager.py   # 统一模型管理器
├── utils/                 # 工具函数
│   ├── helpers.py         # 辅助函数
│   └── middleware.py      # 中间件
├── scripts/               # 脚本文件
│   └── start.sh          # 启动脚本
└── tests/                 # 测试文件
```

### 添加新功能
1. 在 `src/model_manager.py` 中添加新方法
2. 在 `main.py` 中添加新的API路由
3. 更新 `config/settings.py` 中的配置
4. 运行 `python test_project.py` 测试

## 🎊 成功指标

- ✅ Python 3.12环境正常
- ✅ RTX 2060 12GB GPU检测成功
- ✅ 所有依赖安装完成
- ✅ FastAPI应用创建成功
- ✅ 10个API路由正常工作
- ✅ 统一模型架构运行正常

## 📞 支持

如有问题，请检查：
1. 日志文件：`logs/neko_brain.log`
2. 健康检查：`http://localhost:2000/health`
3. 统计信息：`http://localhost:2000/stats`

---

**🎉 恭喜！您的统一NekoBrain项目已成功部署并优化！**
# 统一NekoBrain项目

整合VLM和决策模型为单一服务，针对RTX 2060 12GB优化。

## 项目结构

```
auto_charger/
├── main.py                 # 主应用入口
├── requirements.txt        # Python依赖
├── .env.example           # 环境变量示例
├── __init__.py            # 项目初始化
├── config/                # 配置模块
│   └── settings.py        # 应用配置
├── src/                   # 源代码
│   └── model_manager.py   # 统一模型管理器
├── utils/                 # 工具函数
│   ├── helpers.py         # 辅助函数
│   └── middleware.py      # 中间件
├── scripts/               # 脚本文件
│   └── start.sh          # 启动脚本
├── models/                # 模型缓存目录
├── logs/                  # 日志目录
├── cache/                 # 缓存目录
└── tests/                 # 测试文件
```

## 核心特性

- **统一模型**: 使用Qwen2.5-VL-7B-Instruct同时处理VLM和决策任务
- **量化优化**: 4bit量化适配RTX 2060 12GB显存
- **性能优化**: 减少并发线程数，优化内存使用
- **智能路由**: 基于关键词匹配和模型评分的融合决策
- **VLM支持**: 本地图像理解和OCR功能
- **缓存机制**: LRU缓存提升响应速度
- **监控中间件**: 性能监控和资源管理

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env 文件配置API密钥等
```

### 3. 启动服务

```bash
# 使用启动脚本（推荐）
./scripts/start.sh

# 或直接启动
python main.py
```

### 4. 访问服务

- 服务地址: http://localhost:2000
- 健康检查: http://localhost:2000/health
- API文档: http://localhost:2000/docs

## API接口

### 聊天完成

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

### 健康检查

```bash
curl http://localhost:2000/health
```

### 获取统计信息

```bash
curl http://localhost:2000/stats
```

## 配置说明

### 模型配置

- `UNIFIED_MODEL_ID`: 统一模型ID（默认：Qwen/Qwen2.5-VL-7B-Instruct）
- `LOAD_IN_4BIT`: 启用4bit量化
- `MAX_MEMORY`: 显存限制（默认：10GB）

### 性能配置

- `MAX_WORKERS`: 并发线程数（默认：2）
- `MAX_NEW_TOKENS`: 最大生成token数（默认：80）
- `ROUTE_CACHE_SIZE`: 路由缓存大小（默认：256）

### API配置

- `AGGREGATOR_API_KEY`: 聚合API密钥
- `AGGREGATOR_BASE_URL`: 聚合API地址

## 路由类别

系统支持6种路由类别：

1. **flash_smart**: 通用聊天、问候、简单问题
2. **pro_advanced**: 复杂分析、创意写作、详细解释
3. **code_technical**: 编程、调试、SQL查询、代码编写
4. **code_architect**: 系统设计、软件架构、技术概念解释
5. **logic_reasoning**: 数学证明、物理问题、逻辑推理
6. **expert_xhigh**: 专业研究、学术论文、高上下文分析

## 性能优化

### RTX 2060 12GB优化

- 4bit量化减少显存使用到4-6GB
- 限制并发线程数为2避免显存溢出
- 启用KV cache提升推理速度
- 使用torch.compile加速（如果支持）

### 缓存策略

- LRU缓存存储路由决策结果
- 快速路径关键词匹配
- 图像处理结果缓存

## 监控和调试

### 日志配置

- 日志级别可通过`LOG_LEVEL`设置
- 日志文件位置：`./logs/neko_brain.log`
- 性能日志可开启`NEKOBRAIN_DEBUG=true`

### 性能监控

- `/stats`接口提供缓存和性能统计
- 中间件自动监控CPU、内存、GPU使用情况
- 慢请求自动记录和警告

## 故障排除

### 常见问题

1. **显存不足**: 检查GPU显存，确保至少有10GB可用
2. **模型加载失败**: 检查网络连接，确保能下载模型
3. **端口占用**: 修改PORT配置或终止占用进程
4. **依赖缺失**: 运行`pip install -r requirements.txt`

### 调试模式

设置环境变量启用调试模式：

```bash
export NEKOBRAIN_DEBUG=true
python main.py
```

## 许可证

MIT License

## 更新日志

### v2.0.0

- 整合VLM和决策模型为统一服务
- 针对RTX 2060 12GB优化
- 添加性能监控和中间件
- 改进项目结构和配置管理
- 添加启动脚本和文档
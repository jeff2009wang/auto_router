#!/bin/bash

# 统一NekoBrain启动脚本
# 针对RTX 2060 12GB优化的启动配置

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查Python版本
check_python() {
    log_step "检查Python版本..."
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    log_info "Python版本: $python_version"
    
    # 使用更可靠的版本比较方法
    major_version=$(echo $python_version | cut -d. -f1)
    minor_version=$(echo $python_version | cut -d. -f2)
    
    if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 8 ]]; then
        log_error "需要Python 3.8或更高版本，当前版本: $python_version"
        exit 1
    fi
}

# 检查CUDA
check_cuda() {
    log_step "检查CUDA环境..."
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        log_info "检测到GPU: $gpu_info"
        
        # 检查显存是否足够（至少10GB）
        memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [[ $memory_total -lt 10000 ]]; then
            log_warn "GPU显存可能不足，建议至少10GB显存"
        fi
    else
        log_warn "未检测到NVIDIA GPU，将使用CPU模式"
    fi
}

# 创建必要的目录
setup_directories() {
    log_step "创建必要目录..."
    mkdir -p logs
    mkdir -p models
    mkdir -p cache
    log_info "目录创建完成"
}

# 安装依赖
install_dependencies() {
    log_step "安装Python依赖..."
    
    if [[ -f "requirements.txt" ]]; then
        pip3 install -r requirements.txt
        log_info "依赖安装完成"
    else
        log_error "requirements.txt文件不存在"
        exit 1
    fi
}

# 设置环境变量
setup_environment() {
    log_step "设置环境变量..."
    
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export CUDA_VISIBLE_DEVICES="0"
    export NEKOBRAIN_DEBUG="${NEKOBRAIN_DEBUG:-false}"
    
    log_info "环境变量设置完成"
}

# 启动服务
start_service() {
    log_step "启动统一NekoBrain服务..."
    
    # 检查端口是否被占用
    if lsof -Pi :2000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "端口2000已被占用，尝试终止现有进程..."
        lsof -ti:2000 | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # 启动服务
    log_info "🚀 启动服务在 http://0.0.0.0:2000"
    log_info "📊 健康检查: http://0.0.0.0:2000/health"
    log_info "📖 API文档: http://0.0.0.0:2000/docs"
    log_info "🛑 按Ctrl+C停止服务"
    
    # 使用uvicorn启动
    python3 -m uvicorn main:app \
        --host 0.0.0.0 \
        --port 2000 \
        --reload \
        --log-level info \
        --access-log
}

# 清理函数
cleanup() {
    log_info "正在清理..."
    # 这里可以添加清理逻辑
}

# 信号处理
trap cleanup EXIT
trap 'log_info "收到中断信号，正在退出..."; exit 0' INT TERM

# 主函数
main() {
    echo "=========================================="
    echo "🧠 统一NekoBrain启动脚本"
    echo "📦 版本: 2.0.0"
    echo "🎯 模型: Qwen2.5-VL-7B-Instruct"
    echo "💾 优化: RTX 2060 12GB"
    echo "=========================================="
    
    check_python
    check_cuda
    setup_directories
    setup_environment
    
    # 询问是否安装依赖
    read -p "是否安装依赖? (y/N): " install_deps
    if [[ $install_deps =~ ^[Yy]$ ]]; then
        install_dependencies
    fi
    
    # 询问是否启动服务
    read -p "是否启动服务? (Y/n): " start_now
    if [[ ! $start_now =~ ^[Nn]$ ]]; then
        start_service
    else
        log_info "手动启动命令: python3 -m uvicorn main:app --host 0.0.0.0 --port 2000"
    fi
}

# 运行主函数
main "$@"
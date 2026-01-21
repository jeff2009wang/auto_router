#!/usr/bin/env python3
"""
测试脚本 - 验证项目结构
"""
import sys
import os

sys.path.append('.')

def test_imports():
    """测试模块导入"""
    print("🧪 开始测试模块导入...")
    
    try:
        from config.settings import settings
        print("✅ 配置模块导入成功")
        print(f"   模型: {settings.UNIFIED_MODEL_ID}")
        
        from src.model_manager import UnifiedModelManager
        print("✅ 模型管理器导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fastapi_app():
    """测试FastAPI应用"""
    print("\n🧪 测试FastAPI应用...")
    
    try:
        from main import app
        print("✅ FastAPI应用创建成功")
        print(f"   应用标题: {app.title}")
        print(f"   应用版本: {app.version}")
        return True
        
    except Exception as e:
        print(f"❌ FastAPI应用测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧠 统一NekoBrain项目测试")
    print("=" * 60)
    
    imports_ok = test_imports()
    app_ok = test_fastapi_app()
    
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print(f"   模块导入: {'✅ 通过' if imports_ok else '❌ 失败'}")
    print(f"   FastAPI应用: {'✅ 通过' if app_ok else '❌ 失败'}")
    
    if imports_ok and app_ok:
        print("\n🎉 所有测试通过！项目结构规范完成。")
        print("\n🚀 启动命令:")
        print("   python main.py")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

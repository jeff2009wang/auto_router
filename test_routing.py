"""
路由决策测试脚本
测试优化后的关键词匹配和路由决策逻辑
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings


class MockModelManager:
    """模拟模型管理器用于测试"""
    
    def __init__(self):
        self.logger = None
        self.route_cache = None
        self.full_labels = list(settings.MODEL_MAP.keys())
    
    def _quick_keyword_match(self, text: str):
        """增强的快速关键词匹配 - 支持权重、短语匹配和上下文感知"""
        text_lower = text.lower()
        text_words = text_lower.split()
        text_length = len(text_lower)
        
        # 加权关键词配置
        weighted_keywords = {
            'code_technical': {
                'high_weight': ['def ', 'class ', 'function(', 'import ', 'from ', 'sql ', 'query(', 'debug'],
                'medium_weight': ['python', 'javascript', 'java', 'c++', 'typescript', 'go', 'rust', '代码', '编程', 'api', 'framework', 'library'],
                'low_weight': ['algorithm', 'data structure', 'module', 'package', 'variable', 'array', 'list', 'dictionary']
            },
            'code_architect': {
                'high_weight': ['architecture', 'design pattern', 'system design', 'microservice', '架构', '设计模式', 'scalability'],
                'medium_weight': ['api design', 'database design', 'performance optimization', 'concurrency', 'distributed system', 'restful', 'graphql'],
                'low_weight': ['pattern', 'principle', 'solid', 'dry', 'clean code', 'refactoring']
            },
            'logic_reasoning': {
                'high_weight': ['prove', 'theorem', 'calculate', 'solve(', 'equation', 'integral', '微分', '积分', '证明', '推导'],
                'medium_weight': ['mathematical', 'proof', 'derivation', 'formula', 'probability', 'statistics', 'logic', 'algorithmic'],
                'low_weight': ['math', 'calculation', 'number', 'value', 'result']
            },
            'pro_advanced': {
                'high_weight': ['creative writing', 'story', 'poem', 'novel', '创作', '故事', '诗歌', '小说', 'essay'],
                'medium_weight': ['analysis', 'detailed explanation', 'comprehensive', 'in-depth', 'review', 'critique', 'interpretation'],
                'low_weight': ['write', 'describe', 'explain', 'discuss', 'analyze']
            },
            'flash_smart': {
                'high_weight': ['hello', 'hi ', 'hey', '你好', '谢谢', 'thanks', 'good morning', 'good evening'],
                'medium_weight': ['how are you', 'what is', 'tell me about', 'help me', 'can you'],
                'low_weight': ['question', 'answer', 'ask', 'say', 'tell']
            },
            'expert_xhigh': {
                'high_weight': ['research paper', 'academic', 'thesis', 'dissertation', '研究', '学术', '论文', '文献综述'],
                'medium_weight': ['methodology', 'hypothesis', 'empirical', 'theoretical framework', 'peer review', 'citation', 'journal'],
                'low_weight': ['study', 'analysis', 'investigation', 'experiment', 'data']
            }
        }
        
        # 否定关键词
        negative_keywords = {
            'code_technical': ['hello', 'how are you', 'what is your name', 'tell me a joke', 'story'],
            'pro_advanced': ['code', 'function', 'class', 'debug', 'calculate', 'prove'],
            'flash_smart': ['function', 'class ', 'import ', 'algorithm', 'architecture', 'theorem', 'research paper']
        }
        
        scores = {}
        
        for label, weight_groups in weighted_keywords.items():
            total_score = 0.0
            match_count = 0
            
            for kw in weight_groups['high_weight']:
                if kw in text_lower:
                    if kw.strip() in text_words or ' ' in kw:
                        total_score += 3.0
                        match_count += 1
                    else:
                        total_score += 2.0
                        match_count += 1
            
            for kw in weight_groups['medium_weight']:
                if kw in text_lower:
                    total_score += 2.0
                    match_count += 1
            
            for kw in weight_groups['low_weight']:
                if kw in text_lower:
                    total_score += 1.0
                    match_count += 1
            
            if label in negative_keywords:
                for neg_kw in negative_keywords[label]:
                    if neg_kw in text_lower:
                        total_score -= 2.0
            
            if text_length > 100 and label != 'flash_smart':
                total_score *= 1.1
            
            if text_length < 30 and label in ['code_architect', 'expert_xhigh']:
                total_score *= 0.5
            
            if match_count > 0 and total_score > 0:
                scores[label] = total_score
        
        if scores:
            best_label = max(scores, key=scores.get)
            if scores[best_label] >= 3.0:
                return best_label, scores[best_label]
        
        return None, 0.0


def test_routing():
    """测试路由决策"""
    manager = MockModelManager()
    
    test_cases = [
        ("Write a Python function to calculate fibonacci numbers", "code_technical"),
        ("def hello(): print('world')", "code_technical"),
        ("Design a microservice architecture", "code_architect"),
        ("Prove that square root of 2 is irrational", "logic_reasoning"),
        ("Write a creative story about a robot", "pro_advanced"),
        ("Conduct a literature review on deep learning", "expert_xhigh"),
        ("Hello, how are you today?", "flash_smart"),
        ("你好，很高兴认识你", "flash_smart"),
    ]
    
    print("=" * 80)
    print("路由决策测试结果")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, (text, expected) in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{len(test_cases)}")
        print(f"输入: {text}")
        print(f"期望: {expected}")
        
        quick_label, quick_score = manager._quick_keyword_match(text)
        print(f"快速路径: {quick_label} (score: {quick_score:.1f})")
        
        if quick_label == expected:
            print("✅ 通过")
            passed += 1
        else:
            print("❌ 失败")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"测试总结: {passed} 通过, {failed} 失败")
    print("=" * 80)
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = test_routing()
    sys.exit(0 if failed == 0 else 1)

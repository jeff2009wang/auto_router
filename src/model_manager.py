"""
统一模型管理器
整合VLM和决策模型为单一Qwen2.5-VL-3B模型
"""
import logging
import torch
import time
import hashlib
import asyncio
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from modelscope import snapshot_download
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    BitsAndBytesConfig
)
from PIL import Image
import base64
import io

from config.settings import settings


class LRUCache:
    """LRU缓存实现，用于路由结果缓存"""
    def __init__(self, max_size: int = 256):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Tuple[str, List[Dict]]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Tuple[str, List[Dict]]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()


class UnifiedModelManager:
    """统一模型管理器 - 整合VLM和决策功能"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 性能优化配置
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
        self.enable_perf_logging = True
        
        # 路由缓存
        self.route_cache = LRUCache(max_size=settings.ROUTE_CACHE_SIZE)
        self.full_labels = list(settings.MODEL_MAP.keys())
        
        # 内存管理
        self.inference_count = 0
        self.last_memory_cleanup = 0
        
        # 模型组件
        self.model = None
        self.processor = None
        
        self.logger.info(f"🧠 Initializing Unified Model Manager...")
        self.logger.info(f"🎯 Target Model: {settings.UNIFIED_MODEL_ID}")
        self.logger.info(f"💾 Device: {self.device}")
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, settings.LOG_LEVEL),
            format=settings.LOG_FORMAT
        )
        
        # 抑制第三方库日志
        logging.getLogger("litellm").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        
    def load_model(self):
        """加载统一模型"""
        try:
            self.logger.info("🧠 Loading Unified Qwen2.5-VL-3B Model...")
            
            # 下载模型
            model_dir = snapshot_download(settings.UNIFIED_MODEL_ID)
            
            # 4bit量化配置 - 更激进的内存优化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                bnb_4bit_use_fp16=False  # 禁用FP16以节省内存
            )
            
            # 加载模型 - 使用更激进的内存优化
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: "6GB", "cpu": "16GB"},
                offload_folder=settings.OFFLOAD_FOLDER,
                offload_state_dict=settings.OFFLOAD_STATE_DICT,
                low_cpu_mem_usage=settings.LOW_CPU_MEM_USAGE,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            # 加载处理器
            self.processor = Qwen2_5_VLProcessor.from_pretrained(model_dir, trust_remote_code=True)
            
            # 尝试使用torch.compile加速
            if settings.ENABLE_TORCH_COMPILE and hasattr(torch, 'compile') and self.device == "cuda":
                try:
                    self.logger.info("⚡ Using torch.compile for optimization...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception as e:
                    self.logger.warning(f"torch.compile not available: {e}")
            
            self.logger.info("✅ Unified model loaded successfully")
            self._warmup_model()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise e
    
    def _warmup_model(self):
        """模型预热"""
        self.logger.info("🔥 Warming up model...")
        try:
            dummy_text = "Hello, this is a warmup test."
            self._get_router_scores(dummy_text)
            self.logger.info("✅ Model warmup complete")
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}")
    
    def _decode_base64_image(self, base64_string: str) -> Optional[Image.Image]:
        """解码base64图像"""
        try:
            image_data = base64.b64decode(base64_string)
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            self.logger.error(f"Failed to decode image: {e}")
            return None
    
    def _process_messages_for_vlm(self, messages: List[Dict]) -> Tuple[List[Dict], Optional[Image.Image]]:
        """处理消息，提取图像和文本"""
        processed_messages = []
        extracted_image = None
        
        for message in messages:
            new_message = message.copy()
            content = message.get("content", "")
            
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            new_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") in ["image", "image_url"]:
                            # 处理图像
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                image_url = image_url.get("url", "")
                            elif isinstance(image_url, str):
                                pass  # 已经是字符串
                            else:
                                image_url = str(image_url)
                                
                            if image_url.startswith("data:image"):
                                # Base64编码的图像
                                base64_string = image_url.split(",")[1]
                                extracted_image = self._decode_base64_image(base64_string)
                                if extracted_image:
                                    new_content.append({"type": "image", "image": extracted_image})
                            elif image_url.startswith("http"):
                                # URL图像（简化处理）
                                self.logger.warning("URL images not supported")
                            else:
                                # 假设是base64字符串
                                extracted_image = self._decode_base64_image(image_url)
                                if extracted_image:
                                    new_content.append({"type": "image", "image": extracted_image})
                        else:
                            new_content.append(item)
                    else:
                        # 如果item不是字典，直接添加
                        new_content.append(item)
                new_message["content"] = new_content
            else:
                new_message["content"] = content
                
            processed_messages.append(new_message)
        
        return processed_messages, extracted_image
    
    def _unified_vlm_inference(self, messages: List[Dict], prompt_text: str = None) -> str:
        """使用统一模型进行VLM推理"""
        try:
            self.logger.info("👁️ Using Unified VLM for vision understanding...")
            
            # 处理消息，提取图像
            processed_messages, extracted_image = self._process_messages_for_vlm(messages)
            
            if not extracted_image:
                self.logger.warning("No image found in messages")
                return ""
            
            # 构建输入
            if prompt_text is None:
                prompt_text = "请详细描述这张图片的内容，包括文字、物体、场景等所有可见信息。"
            
            # 准备对话格式
            conversation = []
            for msg in processed_messages:
                role = msg["role"]
                content = msg["content"]
                
                if isinstance(content, list):
                    text_content = ""
                    has_image = False
                    for item in content:
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif item.get("type") == "image":
                            has_image = True
                    if has_image:
                        conversation.append({"role": role, "content": [{"type": "image", "image": extracted_image}, {"type": "text", "text": text_content}]})
                    else:
                        conversation.append({"role": role, "content": text_content})
                else:
                    conversation.append({"role": role, "content": content})
            
            # 添加用户问题
            conversation.append({"role": "user", "content": prompt_text})
            
            # 处理输入
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[extracted_image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # 生成回答
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
            
            # 解码回答
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            self.logger.info("✅ Unified VLM inference complete")
            return output_text
            
        except Exception as e:
            self.logger.error(f"Unified VLM Error: {e}")
            return ""
    
    def _quick_keyword_match(self, text: str) -> Optional[Tuple[str, float]]:
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
        
        # 否定关键词 - 这些词出现时应降低对应类别的分数
        negative_keywords = {
            'code_technical': ['hello', 'how are you', 'what is your name', 'tell me a joke', 'story'],
            'pro_advanced': ['code', 'function', 'class', 'debug', 'calculate', 'prove'],
            'flash_smart': ['function', 'class ', 'import ', 'algorithm', 'architecture', 'theorem', 'research paper']
        }
        
        scores = {}
        
        for label, weight_groups in weighted_keywords.items():
            total_score = 0.0
            match_count = 0
            
            # 检查高权重关键词 (权重: 3.0)
            for kw in weight_groups['high_weight']:
                if kw in text_lower:
                    # 检查是否是完整单词匹配（避免部分匹配）
                    if kw.strip() in text_words or ' ' in kw:
                        total_score += 3.0
                        match_count += 1
                    else:
                        total_score += 2.0
                        match_count += 1
            
            # 检查中权重关键词 (权重: 2.0)
            for kw in weight_groups['medium_weight']:
                if kw in text_lower:
                    total_score += 2.0
                    match_count += 1
            
            # 检查低权重关键词 (权重: 1.0)
            for kw in weight_groups['low_weight']:
                if kw in text_lower:
                    total_score += 1.0
                    match_count += 1
            
            # 应用否定关键词惩罚
            if label in negative_keywords:
                for neg_kw in negative_keywords[label]:
                    if neg_kw in text_lower:
                        total_score -= 2.0  # 惩罚
            
            # 长度奖励：较长的查询更可能需要复杂处理
            if text_length > 100 and label != 'flash_smart':
                total_score *= 1.1
            
            # 短查询惩罚：非常短的查询不太可能是复杂任务
            if text_length < 30 and label in ['code_architect', 'expert_xhigh']:
                total_score *= 0.5
            
            if match_count > 0 and total_score > 0:
                scores[label] = total_score
        
        if scores:
            best_label = max(scores, key=scores.get)
            # 调整阈值：需要至少3分或有2个以上匹配
            if scores[best_label] >= 3.0:
                return best_label, scores[best_label]
        
        return None, 0.0
    
    def _normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """归一化分数"""
        scores = list(raw_scores.values())
        if not scores:
            return raw_scores
        
        min_score, max_score = min(scores), max(scores)
        
        if max_score == min_score:
            return {label: 5.0 for label in raw_scores.keys()}
        
        return {
            label: 1.0 + 9.0 * (score - min_score) / (max_score - min_score)
            for label, score in raw_scores.items()
        }
    
    def _improve_routing_decision(self, router_scores: Dict[str, float], text: str) -> Dict[str, float]:
        """增强的路由决策逻辑 - 多维度分析、动态权重调整"""
        # 首先归一化分数
        normalized_scores = self._normalize_scores(router_scores)
        
        # 文本分析
        text_lower = text.lower()
        text_length = len(text)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
        
        # 计算文本特征
        avg_word_length = sum(len(word) for word in text_lower.split()) / max(word_count, 1)
        complexity_score = (text_length / 100) * (avg_word_length / 5) * (word_count / 20)
        
        # 扩展的关键词检测 - 支持多语言
        category_keywords = {
            'code_technical': {
                'programming': ['def ', 'class ', 'function(', 'import ', 'from ', 'lambda', 'async', 'await'],
                'languages': ['python', 'javascript', 'java', 'c++', 'typescript', 'go', 'rust', 'swift', 'kotlin'],
                'concepts': ['algorithm', 'data structure', 'recursion', 'iteration', 'string manipulation', 'regex', 'api', 'rest', 'graphql'],
                'chinese': ['代码', '编程', '函数', '类', '调试', '算法', '数据结构', '接口'],
                'actions': ['debug', 'implement', 'refactor', 'optimize', 'write', 'create', 'build']
            },
            'code_architect': {
                'high_level': ['architecture', 'design pattern', 'system design', 'microservice', 'monolith', 'serverless'],
                'principles': ['scalability', 'maintainability', 'reliability', 'performance', 'security', 'coupling', 'cohesion'],
                'patterns': ['singleton', 'factory', 'observer', 'strategy', 'decorator', 'adapter', 'mvc', 'mvvm'],
                'chinese': ['架构', '设计模式', '可扩展性', '微服务', '分布式', '高并发', '解耦'],
                'areas': ['database design', 'api design', 'system integration', 'cloud architecture', 'devops']
            },
            'logic_reasoning': {
                'math': ['prove', 'theorem', 'lemma', 'corollary', 'calculate', 'solve', 'equation', 'inequality', 'integral', 'derivative'],
                'logic': ['if and only if', 'therefore', 'hence', 'implies', 'contradiction', 'induction', 'deduction'],
                'chinese': ['证明', '定理', '推导', '微分', '积分', '方程', '不等式', '归纳'],
                'topics': ['probability', 'statistics', 'combinatorics', 'graph theory', 'number theory', 'geometry', 'calculus'],
                'actions': ['calculate', 'compute', 'determine', 'find', 'solve', 'derive']
            },
            'pro_advanced': {
                'creative': ['story', 'poem', 'novel', 'essay', 'article', 'blog post', 'creative writing', 'narrative'],
                'analysis': ['analyze', 'critique', 'review', 'evaluate', 'assess', 'interpret', 'examine'],
                'chinese': ['创作', '故事', '诗歌', '小说', '分析', '评论', '解读', '深度'],
                'requirements': ['detailed explanation', 'comprehensive', 'in-depth', 'thorough', 'extensive', 'nuanced'],
                'domains': ['literature', 'philosophy', 'psychology', 'sociology', 'cultural studies', 'history']
            },
            'expert_xhigh': {
                'academic': ['research', 'paper', 'journal', 'conference', 'publication', 'citation', 'bibliography'],
                'research': ['methodology', 'hypothesis', 'empirical', 'theoretical', 'qualitative', 'quantitative', 'peer review'],
                'chinese': ['研究', '学术', '论文', '文献', '方法论', '假设', '实证', '理论'],
                'domains': ['machine learning', 'deep learning', 'artificial intelligence', 'data science', 'bioinformatics', 'economics'],
                'tasks': ['literature review', 'meta-analysis', 'systematic review', 'comparative study']
            },
            'flash_smart': {
                'greetings': ['hello', 'hi ', 'hey', 'good morning', 'good evening', 'good afternoon'],
                'chinese': ['你好', '您好', '早上好', '晚上好', '谢谢', '感谢'],
                'casual': ['how are you', 'what\'s up', 'how\'s it going', 'thanks', 'thank you'],
                'simple': ['what is', 'tell me', 'help me', 'can you', 'please', 'question'],
                'social': ['joke', 'funny', 'interesting', 'tell me about', 'what do you think']
            }
        }
        
        # 计算每个类别的关键词匹配分数
        category_match_scores = {}
        for category, keyword_groups in category_keywords.items():
            match_score = 0.0
            for group_name, keywords in keyword_groups.items():
                group_matches = sum(1 for kw in keywords if kw in text_lower)
                if group_matches > 0:
                    # 根据组的重要性加权
                    weight = 1.5 if group_name in ['programming', 'languages', 'chinese', 'high_level', 'math', 'creative', 'academic', 'greetings'] else 1.0
                    match_score += group_matches * weight
            category_match_scores[category] = match_score
        
        # 调整分数
        adjusted_scores = normalized_scores.copy()
        
        # 1. 基于复杂度调整
        if complexity_score > 2.0:
            adjusted_scores['flash_smart'] *= 0.3
            adjusted_scores['pro_advanced'] *= 1.3
            adjusted_scores['expert_xhigh'] *= 1.2
        elif complexity_score > 1.0:
            adjusted_scores['flash_smart'] *= 0.5
            adjusted_scores['pro_advanced'] *= 1.1
        
        # 2. 基于关键词匹配调整
        max_match_score = max(category_match_scores.values()) if category_match_scores else 0
        if max_match_score > 0:
            for category, match_score in category_match_scores.items():
                if match_score > 0:
                    # 匹配分数越高，提升幅度越大
                    boost_factor = 1.0 + (match_score / max_match_score) * 0.5
                    adjusted_scores[category] *= boost_factor
                    
                    # 同时抑制其他类别
                    if category != 'flash_smart':
                        adjusted_scores['flash_smart'] *= 0.8
        
        # 3. 特殊规则调整
        # 代码相关：如果有多个代码关键词，强烈倾向于code类别
        if category_match_scores['code_technical'] >= 3 or category_match_scores['code_architect'] >= 2:
            adjusted_scores['code_technical'] *= 1.8
            adjusted_scores['code_architect'] *= 1.6
            adjusted_scores['flash_smart'] *= 0.2
            adjusted_scores['pro_advanced'] *= 0.6
        
        # 数学/逻辑相关
        if category_match_scores['logic_reasoning'] >= 2:
            adjusted_scores['logic_reasoning'] *= 2.0
            adjusted_scores['flash_smart'] *= 0.3
            adjusted_scores['pro_advanced'] *= 0.7
        
        # 学术研究相关
        if category_match_scores['expert_xhigh'] >= 2:
            adjusted_scores['expert_xhigh'] *= 2.2
            adjusted_scores['flash_smart'] *= 0.1
            adjusted_scores['pro_advanced'] *= 1.2
        
        # 创意写作相关
        if category_match_scores['pro_advanced'] >= 2:
            adjusted_scores['pro_advanced'] *= 1.8
            adjusted_scores['flash_smart'] *= 0.5
        
        # 4. 上下文感知调整
        # 如果文本很短且没有复杂关键词，倾向于flash_smart
        if text_length < 50 and max_match_score < 2:
            adjusted_scores['flash_smart'] *= 2.0
            for category in ['code_technical', 'code_architect', 'logic_reasoning', 'expert_xhigh']:
                adjusted_scores[category] *= 0.3
        
        # 如果文本非常长，倾向于高级分析
        if text_length > 500:
            adjusted_scores['pro_advanced'] *= 1.3
            adjusted_scores['expert_xhigh'] *= 1.2
            adjusted_scores['flash_smart'] *= 0.4
        
        # 5. 防止分数过低
        min_score = 0.5
        for label in adjusted_scores:
            adjusted_scores[label] = max(min_score, adjusted_scores[label])
        
        # 6. 最终归一化，确保分数在合理范围内
        total = sum(adjusted_scores.values())
        if total > 0:
            target_total = len(adjusted_scores) * 5.0  # 平均分约5
            scale_factor = target_total / total
            # 限制缩放因子，避免过度放大
            scale_factor = max(0.5, min(2.0, scale_factor))
            adjusted_scores = {k: v * scale_factor for k, v in adjusted_scores.items()}
        
        return adjusted_scores
    
    @torch.no_grad()
    def _get_router_scores(self, text: str) -> Dict[str, float]:
        """使用统一模型进行路由决策"""
        start_time = time.time()
        try:
            context_segment = text[:800]
            
            prompt = (
                "Rate the user input for EACH category below. You MUST rate ALL 6 categories.\n"
                "Score: 1 = Not relevant, 10 = Perfect match\n\n"
                "Categories:\n"
                "1. flash_smart: General chat, greetings, simple questions, daily conversation\n"
                "2. pro_advanced: Complex analysis, creative writing, nuanced language understanding, detailed explanations\n"
                "3. code_technical: Programming, debugging, SQL queries, writing code in Python/C++/Java, technical scripts\n"
                "4. code_architect: System design, software architecture, explaining technical concepts, architectural patterns\n"
                "5. logic_reasoning: Math proofs, physics problems, logic puzzles, step-by-step reasoning, calculus, theorems\n"
                "6. expert_xhigh: Professional research, academic papers, high-context analysis, specialized knowledge\n\n"
                f"User Input: \"{context_segment}\"\n\n"
                "Output ALL 6 ratings in format: label:X (one per line, where X is a number from 1 to 10)."
            )
            
            messages = [
                {"role": "system", "content": "You are a precise classifier. Rate each category from 1 to 10 based on relevance."},
                {"role": "user", "content": prompt}
            ]
            
            # 使用统一模型进行路由决策
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.processor(text=[text_input], return_tensors="pt").to(self.model.device)
            
            # 生成路由决策
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=settings.ENABLE_KV_CACHE
            )
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 解析评分
            scores = {}
            for line in response.strip().split('\n'):
                line = line.strip()
                if ':' not in line:
                    continue
                
                for separator in [':', '=', ' ']:
                    if separator in line:
                        parts = line.split(separator, 1)
                        if len(parts) == 2:
                            potential_label = parts[0].strip().lower()
                            potential_score = parts[1].strip()
                            
                            for label in self.full_labels:
                                if label.lower() in potential_label or potential_label in label.lower():
                                    score_str = ""
                                    for char in potential_score:
                                        if char.isdigit() or char == '.':
                                            score_str += char
                                        elif char in [' ', '\t'] and score_str:
                                            break
                                        elif char not in [' ', '\t'] and not (char.isdigit() or char == '.'):
                                            if score_str:
                                                break
                                    
                                    if score_str:
                                        try:
                                            score = float(score_str)
                                            if 0 <= score <= 10:
                                                scores[label] = score
                                                break
                                        except ValueError:
                                            continue
            
            for label in self.full_labels:
                if label not in scores:
                    scores[label] = 1.0
            
            if self.enable_perf_logging:
                self.logger.info(f"⚡ Router: {(time.time() - start_time)*1000:.1f}ms")
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Router scoring error: {e}")
            return {label: 1.0 for label in self.full_labels}
    
    def _get_text_hash(self, text: str) -> str:
        """生成文本hash用于缓存"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _cleanup_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if settings.DEBUG:
                self.logger.info(f"🧹 Memory cleaned. GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    def _check_memory_cleanup(self):
        """检查是否需要清理内存"""
        self.inference_count += 1
        if (self.inference_count - self.last_memory_cleanup) >= settings.MEMORY_CLEANUP_INTERVAL:
            self._cleanup_memory()
            self.last_memory_cleanup = self.inference_count
    
    def _get_fused_decision(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        """融合决策逻辑"""
        decision_start = time.time()
        target_text = ""
        modified_messages = messages
        
        # 检查内存清理
        self._check_memory_cleanup()
        
        # 检查是否有图像
        has_image = any(
            isinstance(m.get("content"), list) and any(item.get("type") in ["image", "image_url"] for item in m["content"])
            for m in messages[-2:]
        )
        
        if has_image:
            self.logger.info("📸 Image detected. Starting Unified VLM processing...")
            try:
                extracted_text = self._unified_vlm_inference(messages)
                target_text = extracted_text
                modified_messages = []
                for m in messages:
                    new_m = m.copy()
                    if isinstance(new_m.get("content"), list):
                        new_m["content"] = f"【System Note: Image Content (OCR):】\n{extracted_text}"
                    modified_messages.append(new_m)
            except Exception as e:
                self.logger.error(f"VLM processing failed: {e}")
                # 如果VLM失败，回退到文本路由
                has_image = False
        
        if not has_image:
            last_msg = messages[-1]
            if isinstance(last_msg["content"], str):
                target_text = last_msg["content"]
            elif isinstance(last_msg["content"], list):
                for item in last_msg["content"]:
                    if item.get("type") == "text":
                        target_text += item.get("text", "")
        
        # 检查缓存
        if not has_image and target_text:
            text_hash = self._get_text_hash(target_text)
            cached_result = self.route_cache.get(text_hash)
            if cached_result:
                self.logger.info(f"⚡ Cache hit! Route: {cached_result[0]} ({((time.time() - decision_start)*1000):.1f}ms)")
                return cached_result
        
        # 快速路径
        if target_text and len(target_text) < 500:
            quick_label, quick_score = self._quick_keyword_match(target_text)
            if quick_label:
                self.logger.info(f"⚡ Quick path: {quick_label} (score: {quick_score:.1f}) ({((time.time() - decision_start)*1000):.1f}ms)")
                result = (quick_label, modified_messages)
                if not has_image and target_text:
                    text_hash = self._get_text_hash(target_text)
                    self.route_cache.put(text_hash, result)
                return result
        
        # 使用统一模型进行路由决策
        try:
            router_scores = self._get_router_scores(target_text)
            
            # 改进的路由决策逻辑
            final_scores = self._improve_routing_decision(router_scores, target_text)
            best_label = max(final_scores, key=final_scores.get)
            
            # Debug模式：显示详细评分
            if settings.DEBUG:
                self.logger.info(f"🔍 Debug - Raw scores: {router_scores}")
                self.logger.info(f"🔍 Debug - Final scores: {final_scores}")
                self.logger.info(f"🔍 Debug - Best label: {best_label}")
            
            # 缓存结果
            if not has_image and target_text:
                result = (best_label, modified_messages)
                text_hash = self._get_text_hash(target_text)
                self.route_cache.put(text_hash, result)
            
            # 日志输出
            if self.enable_perf_logging:
                self.logger.info(f"🎯 Route: {best_label} ({((time.time() - decision_start)*1000):.1f}ms)")
            
            return best_label, modified_messages
            
        except Exception as e:
            self.logger.error(f"Routing decision failed: {e}")
            # 回退到默认路由
            return "flash_smart", modified_messages
    
    async def route(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        """异步路由接口"""
        return await asyncio.get_event_loop().run_in_executor(self.executor, self._get_fused_decision, messages)
    
    def inject_assistant_prompt(self, messages: List[Dict]) -> List[Dict]:
        """注入助手提示"""
        new_msgs = [m.copy() for m in messages]
        injection = {
            "role": "assistant",
            "content": "I will provide a professional solution. For code, I will optimize it. For math, I use LaTeX.\n"
        }
        new_msgs.append(injection)
        return new_msgs
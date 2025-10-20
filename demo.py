import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
import random
from PIL import Image
import torch
from torchvision import models, transforms
import time
from openai import OpenAI

# --------------------------
# 1. LLM配置（DeepSeek）
# --------------------------
class LLMClient:
    def __init__(self):
        self.base_url = "https://api.deepseek.com"
        self.base_model = "deepseek-chat"
        self.api_key = "sk-b1882ddd341f4871bbcf093efe68ddf5"  # 替换为你的API密钥
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call_llm(self, system_msg, user_msg, retries=2, sleep_between=1.0):
        """通用LLM调用函数，带重试机制"""
        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.base_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.3,  # 低温度保证结果稳定性
                    max_tokens=500,
                    timeout=60
                )
                content = getattr(resp.choices[0].message, "content", "").strip()
                if content:
                    return content
            except Exception as e:
                print(f"LLM调用失败（尝试 {attempt+1}/{retries+1}）：{str(e)}")
                if attempt < retries:
                    time.sleep(sleep_between)
        return None


# --------------------------
# 2. 多模态模型（真实图片处理）
# --------------------------
class RealMultiModalModel:
    """使用ResNet提取图片特征，与文本特征对齐"""
    def __init__(self):
        # 图像特征提取器
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = torch.nn.Identity()  # 输出特征向量
        self.image_model.eval()
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def image_to_vec(self, image_path: str) -> np.ndarray:
        """真实图片转向量"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.image_transform(img).unsqueeze(0)
            with torch.no_grad():
                features = self.image_model(img_tensor)
            return features.numpy().flatten()  # 保留原始维度（2048维）
        except Exception as e:
            print(f"图片处理失败：{e}")
            return np.random.rand(2048)


# --------------------------
# 3. 动态时空记忆库
# --------------------------
class DynamicMemoryBank:
    def __init__(self):
        # 静态基础库：场景和物品属性
        self.static_db = {
            "scenes": {
                "living_room": {"objects": ["sofa", "coffee_table", "tv"], "position": "客厅"},
                "study_room": {"objects": ["desk", "bookshelf", "chair"], "position": "书房"},
                "bedroom": {"objects": ["bed", "nightstand"], "position": "卧室"}
            },
            "items": {
                "blue_cup": {"attributes": ["蓝色", "圆柱形", "用于喝水"], "name": "蓝色水杯"},
                "red_book": {"attributes": ["红色", "长方形", "用于阅读"], "name": "红色的书"}
            }
        }
        
        # 动态时序库：物品-位置-时间-视觉证据
        self.dynamic_db = [
            {
                "item": "blue_cup",
                "location": "living_room.coffee_table",
                "timestamp": datetime.now() - timedelta(hours=2),
                "image_path": "images/blue_cup_living.jpg",
                "source": "robot_sensor",
                "confidence": 0.85
            },
            {
                "item": "blue_cup",
                "location": "study_room.desk",
                "timestamp": datetime.now() - timedelta(hours=0.5),
                "image_path": "images/blue_cup_study.jpg",
                "source": "robot_sensor",
                "confidence": 0.92
            },
            {
                "item": "red_book",
                "location": "study_room.bookshelf",
                "timestamp": datetime.now() - timedelta(hours=1),
                "image_path": "images/red_book.jpg",
                "source": "user_input",
                "confidence": 0.98
            }
        ]
    
    def update_memory(self, new_record: dict):
        """更新动态记忆库"""
        self.dynamic_db.append(new_record)
        print(f"\n记忆库已更新：{new_record['item']} 在 {new_record['location']}（{new_record['timestamp']}）")


# --------------------------
# 4. RAG核心检索模块（整合LLM）
# --------------------------
class RAGRetriever:
    def __init__(self, memory_bank: DynamicMemoryBank, multimodal_model: RealMultiModalModel, llm_client: LLMClient):
        self.memory = memory_bank
        self.model = multimodal_model
        self.llm = llm_client  # LLM客户端
    
    def parse_user_intent(self, user_query: str) -> dict:
        """用LLM解析用户意图（替代硬编码规则）"""
        system_msg = """
        你是机器人的意图解析模块。请分析用户查询，提取三个关键信息：
        1. 目标物品（item）：用户要找的物品名称（如“蓝色水杯”对应内部标识“blue_cup”，“红色的书”对应“red_book”）
        2. 时间约束（time_constraint）：用户是否指定时间（如“早上”“昨天”，默认“recent”表示最近）
        3. 位置约束（location_constraint）：用户是否指定位置（如“客厅”，默认“none”）
        
        输出格式为JSON，例如：
        {"item": "blue_cup", "time_constraint": "recent", "location_constraint": "none"}
        若无法识别，item设为null。
        """
        user_msg = f"解析用户查询：{user_query}"
        
        # 调用LLM解析意图
        llm_response = self.llm.call_llm(system_msg, user_msg)
        if not llm_response:
            return {"item": None, "time_constraint": "recent", "location_constraint": "none"}
        
        # 解析JSON结果
        try:
            import json
            return json.loads(llm_response)
        except:
            print(f"意图解析格式错误：{llm_response}")
            return {"item": None, "time_constraint": "recent", "location_constraint": "none"}
    
    def retrieve_candidates(self, intent: dict) -> list:
        """粗检索：按物品和约束筛选"""
        candidates = []
        target_item = intent["item"]
        if not target_item:
            return []
        
        # 筛选同物品记录
        for record in self.memory.dynamic_db:
            if record["item"] == target_item:
                # 时间约束：最近1小时（可扩展为LLM解析的复杂时间）
                if (datetime.now() - record["timestamp"]) < timedelta(hours=1):
                    candidates.append(record)
        return candidates if candidates else [r for r in self.memory.dynamic_db if r["item"] == target_item]
    
    def multimodal_align(self, candidates: list, user_query: str) -> list:
        """多模态对齐：用CLIP逻辑计算相似度（此处简化为特征匹配）"""
        # 实际应用中应使用LLM生成文本向量，这里用随机向量模拟
        intent_vec = np.random.rand(2048)  # 模拟文本向量（真实场景用CLIP文本编码器）
        
        scored_candidates = []
        for cand in candidates:
            image_vec = self.model.image_to_vec(cand["image_path"])
            similarity = cosine_similarity([intent_vec], [image_vec])[0][0]
            final_score = 0.7 * cand["confidence"] + 0.3 * similarity
            scored_candidates.append({**cand, "alignment_score": final_score})
        
        return sorted(scored_candidates, key=lambda x: x["alignment_score"], reverse=True)
    
    def get_best_evidence(self, user_query: str) -> dict:
        """整合检索流程"""
        intent = self.parse_user_intent(user_query)
        candidates = self.retrieve_candidates(intent)
        aligned = self.multimodal_align(candidates, user_query)
        return aligned[0] if aligned else None


# --------------------------
# 5. 机器人决策模块（用LLM生成决策）
# --------------------------
class RobotController:
    def __init__(self, retriever: RAGRetriever, memory_bank: DynamicMemoryBank, llm_client: LLMClient):
        self.retriever = retriever
        self.memory = memory_bank
        self.llm = llm_client
        self.current_location = "living_room"
    
    def visual_perception(self, target_location: str) -> dict:
        """模拟视觉感知（拍摄图片并识别）"""
        print(f"\n机器人移动到 {target_location}，正在拍摄图像...")
        live_image_path = f"images/live_{target_location.replace('.', '_')}.jpg"
        
        # 生成示例图片（实际替换为摄像头拍摄）
        img = Image.new('RGB', (224, 224), color='blue' if "blue_cup" in target_location else 'red')
        img.save(live_image_path)
        
        # 模拟识别结果（70%成功率）
        success = random.choice([True, True, False])
        return {
            "detected": success,
            "description": f"在 {target_location} 检测到目标物品" if success else f"在 {target_location} 未检测到目标物品",
            "image_path": live_image_path if success else None
        }
    
    def make_decision(self, user_query: str):
        """用LLM生成决策：基于检索结果和感知信息"""
        # 1. RAG检索最优证据
        best_evidence = self.retriever.get_best_evidence(user_query)
        if not best_evidence:
            print("未找到相关记忆，调用LLM生成探索方案...")
            self._generate_exploration_plan(user_query)
            return
        
        # 2. 视觉验证
        perception_result = self.visual_perception(best_evidence["location"])
        print(f"视觉验证结果：{perception_result['description']}")
        
        # 3. 用LLM生成决策
        system_msg = """
        你是机器人的决策模块。根据以下信息决定下一步动作：
        - 检索证据：物品历史位置和时间
        - 感知结果：当前位置的视觉检测结果
        
        可能的决策包括：
        1. 若检测成功：确认找到物品，更新记忆库
        2. 若检测失败：建议检索次优证据重新验证，或生成探索路径
        
        输出决策时需具体、可执行（如“前往书房书桌验证”“更新记忆库记录”）。
        """
        user_msg = f"""
        检索证据：{best_evidence['item']} 最近在 {best_evidence['location']}（{best_evidence['timestamp']}）
        感知结果：{perception_result['description']}
        请生成下一步决策。
        """
        
        decision = self.llm.call_llm(system_msg, user_msg)
        print(f"\nLLM生成决策：{decision}")
        
        # 执行决策（实际应根据LLM输出解析执行）
        if perception_result["detected"]:
            new_record = {
                "item": best_evidence["item"],
                "location": best_evidence["location"],
                "timestamp": datetime.now(),
                "image_path": perception_result["image_path"],
                "source": "robot_verified",
                "confidence": 0.99
            }
            self.memory.update_memory(new_record)
        else:
            print("执行次优检索流程...")
    
    def _generate_exploration_plan(self, user_query: str):
        """当无记忆时，用LLM生成探索方案"""
        system_msg = """
        你是机器人的探索规划模块。当没有物品历史记录时，根据用户查询生成探索方案：
        1. 分析物品可能出现的位置（如“水杯”可能在客厅、厨房）
        2. 规划探索顺序（按概率排序）
        3. 每个位置的检查要点（如“检查茶几表面”）
        """
        user_msg = f"用户要找：{user_query}，请生成探索方案（分步骤）。"
        plan = self.llm.call_llm(system_msg, user_msg)
        print(f"探索方案：{plan}")


# --------------------------
# 准备示例图片
# --------------------------
def prepare_demo_images():
    os.makedirs("images", exist_ok=True)
    # 创建示例图片（蓝色水杯和红色书）
    for name in ["blue_cup_living.jpg", "blue_cup_study.jpg", "red_book.jpg"]:
        color = "blue" if "blue" in name else "red"
        img = Image.new('RGB', (224, 224), color=color)
        img.save(f"images/{name}")
    print("示例图片已生成至 images 目录\n")


# --------------------------
# 运行Demo
# --------------------------
if __name__ == "__main__":
    prepare_demo_images()
    
    # 初始化组件
    llm_client = LLMClient()
    multimodal_model = RealMultiModalModel()
    memory_bank = DynamicMemoryBank()
    rag_retriever = RAGRetriever(memory_bank, multimodal_model, llm_client)
    robot = RobotController(rag_retriever, memory_bank, llm_client)
    
    # 用户指令示例（可修改为更复杂的查询）
    user_queries = [
        "帮我找一下我的蓝色水杯",
        "昨天放在书房的红色的书在哪里",
        "我需要那个蓝色的杯子，早上还在客厅"
    ]
    
    # 执行查询
    for query in user_queries[:1]:  # 先运行一个示例
        print(f"\n===== 用户指令：{query} =====")
        robot.make_decision(query)
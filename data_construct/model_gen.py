import os
import re
import json
from openai import OpenAI
import time
import random

# DeepSeek 配置
base_url = "https://api.deepseek.com"
base_model = "deepseek-chat"
api_key = "sk-b1882ddd341f4871bbcf093efe68ddf5"  # 替换为您的API密钥
client = OpenAI(api_key=api_key, base_url=base_url)


def generate_search_scene(
    character_name=None,
    item=None,
    location=None,
    retries=2,
    sleep_between=1.0
):
    """
    调用大模型生成一个寻找场景的文本段落
    """
    # 随机选择一些元素，增加场景多样性
    if not character_name:
        character_names = ["李明", "张华", "王芳", "赵伟", "陈静"]
        character_name = random.choice(character_names)
    
    if not item:
        items = [
            "祖传的青铜钥匙", "外婆留下的玉手镯", "一份重要的合同文件",
            "爷爷的旧怀表", "丢失的护照", "装有秘密的小木盒",
            "童年的日记本", "珍贵的邮票收藏", "一个神秘的包裹",
            "一双旧鞋子", "一把破旧的雨伞", "一个破旧的背包",
            "一副新眼镜", "一个玩具熊", "一个玩具车"
        ]
        item = random.choice(items)
    
    if not location:
        locations = [
            "老式阁楼", "狭窄的储藏室", "杂乱的书房", "尘封的地下室",
            "拥挤的衣帽间", "堆满杂物的车库", "老旧的厨房", "狭小的浴室"
        ]
        location = random.choice(locations)

    # 构建详细的提示词
    system_msg = (
        "你是一位擅长场景描写的作家。请创作一段细腻生动的场景描写，"
        "内容是一个人在密闭空间中寻找某个物品，最终成功找到的过程。"
        "请重点刻画：1) 密闭空间的环境细节和氛围；2) 人物的动作、神态和心理活动；"
        "3) 所寻找物品的特征和细节；4) 找到物品时的情景。"
        "语言要生动具体，有画面感，长度适中，形成一个完整的段落。"
    )
    
    user_msg = (
        f"请创作一个场景：人物{character_name}在{location}中寻找{item}，"
        "最终找到了它。请详细描写空间环境、人物的寻找动作和物品的样子。"
        "确保场景完整，有始有终，细节丰富。"
    )

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=base_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,  # 适当提高温度增加创造性
                max_tokens=800,
                timeout=60
            )
            
            content = getattr(resp.choices[0].message, "content", "") or ""
            if content.strip():
                return {
                    "character": character_name,
                    "item": item,
                    "location": location,
                    "paragraph": content.strip()
                }
                
        except Exception as e:
            print(f"生成失败（尝试 {attempt+1}/{retries+1}）：{str(e)}")
            if attempt < retries:
                time.sleep(sleep_between)
    
    return None


def generate_multiple_scenes(
    num_scenes=5,
    output_path="generated_search_scenes.json"
):
    """
    生成多个寻找场景并保存到JSON文件
    """
    scenes = []
    
    print(f"开始生成 {num_scenes} 个寻找场景...")
    
    for i in range(num_scenes):
        print(f"生成场景 {i+1}/{num_scenes}")
        scene = generate_search_scene()
        
        if scene:
            scenes.append(scene)
            print(f"场景 {i+1} 生成成功")
        else:
            print(f"场景 {i+1} 生成失败")
        
        # 避免请求过于频繁
        if i < num_scenes - 1:
            time.sleep(1.5)
    
    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    
    print(f"生成完成，共成功生成 {len(scenes)} 个场景，已保存到 {output_path}")
    return scenes


if __name__ == "__main__":
    # 生成10个不同的寻找场景
    generate_multiple_scenes(num_scenes=50, output_path="generated_search_scenes_3.json")

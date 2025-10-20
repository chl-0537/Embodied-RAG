# test_chatglm_vision_official.py
import requests
import json

# ----------------- 配置 -----------------
CHATGLM_TOKEN = "fb4968f5f4cc415dbadc110ca418675a.AlJqhOJYMkIi2X0Q"
CHATGLM_VISION_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
VISION_MODEL = "glm-4v-flash"  # 或 "glm-4.5v"

HEADERS = {
    "Authorization": f"Bearer {CHATGLM_TOKEN}",
    "Content-Type": "application/json"
}

def test_chatglm_vision():
    image_urls = [
        "https://known-black-9eebdukx4b.edgeone.app/%E7%BB%98%E5%88%B6%E4%B9%A6%E6%88%BF%203D%20%E5%9B%BE%E5%83%8F.png"
    ]
    question = "What are the pics about?"

    # 构造 messages
    content_list = []
    for u in image_urls:
        content_list.append({"type": "image_url", "image_url": {"url": u}})
    content_list.append({"type": "text", "text": question})

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "user", "content": content_list}
        ]
    }

    try:
        print(payload)
        resp = requests.post(CHATGLM_VISION_URL, headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        print("接口调用成功！返回内容:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("\n模型输出文本:", data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[Error] 调用 ChatGLM 视觉接口失败: {e}")

if __name__ == "__main__":
    test_chatglm_vision()

import base64
from zai import ZhipuAiClient

with open("./picture/study_room.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

client = ZhipuAiClient(api_key="fb4968f5f4cc415dbadc110ca418675a.AlJqhOJYMkIi2X0Q") # 填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4v-flash",  # 填写需要调用的模型名称
    messages=[
       {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "图里有什么"
          },
          {
            "type": "image_url",
            "image_url": {
                "url" : base64_image
            }
          }
        ]
      }
    ],
    temperature=0.5,
    max_tokens=2000,
)
print(response.choices[0].message.content)
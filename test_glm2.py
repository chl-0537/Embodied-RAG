import os
import base64
from mimetypes import guess_type
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv(), verbose=True)

class LLMClient(ChatOpenAI):
    """自定义 GLM-4V 多模态聊天模型类"""
    
    def __init__(self, model = "glm-4v-flash", api_key = "fb4968f5f4cc415dbadc110ca418675a.AlJqhOJYMkIi2X0Q", base_url ="https://open.bigmodel.cn/api/paas/v4/", agent_mode = False, **kwargs):
        super().__init__(api_key=api_key,base_url=base_url,model=model or "GLM-4.1V-Thinking-FlashX", **kwargs)
        # self.agent_mode = agent_mode
        # if self.agent_mode:
        #     self.agent = create_react_agent(api_key=api_key,base_url=base_url,model=model or "GLM-4.1V-Thinking-FlashX", **kwargs)
        # self.openai_api_base = base_url
        # self.openai_api_key = api_key
        # self.last_generated_at = None

        # 覆盖 LangChain 默认配置为 GLM-4V 参数
        # self.model = os.getenv("MODEL")  # GLM-4.1V-Thinking-FlashX
        
    def _encode_image(self, image_path: str) -> str:
        """将本地图像编码为 Data URI 格式"""
        mime_type, _ = guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            raise ValueError(f"Unsupported image type: {image_path}")
        
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
        
        return f"data:{mime_type};base64,{encoded_image}"
    
    def invoke_with_image(
        self,
        prompt: str,
        image_path: str,
        system_prompt: str = "你是一位视觉助手，请仔细分析图像并回答问题",
        **kwargs
    ):
        """
        发送图像+文本的多模态请求
        :param prompt: 用户问题提示
        :param image_path: 本地图像路径
        :param system_prompt: 系统角色设定 (可选)
        :param kwargs: 其他模型参数 (temperature, max_tokens等)
        :return: 模型响应内容
        """
        try:
            # 编码图像为Data URI
            # image_uri = self._encode_image(image_path)
            # 构造多模态消息
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_path}}
                ])
            ]
            
            # 调用模型API
            response = super().invoke(input=messages, **kwargs)
            
            return response.content
        
        except Exception as e:
            raise RuntimeError(f"GLM-4V调用失败: {str(e)}") from e

    def invoke(
        self,
        prompt: str,
        system_prompt: str = "你是一位助手，请根据用户的问题提供准确的回答",
        **kwargs
    ):
        """
        发送文本请求
        :param prompt: 用户问题提示
        :param system_prompt: 系统角色设定 (可选)
        :param kwargs: 其他模型参数 (temperature, max_tokens等)
        :return: 模型响应内容
        """
        try:
            # 构造文本消息
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # 调用模型API
            response = super().invoke(input=messages, **kwargs)
            return response.content
        
        except Exception as e:
            raise RuntimeError(f"OpenAI调用失败: {str(e)}") from e


# 单独执行本文件的测试代码
if __name__ == "__main__":
    client = LLMClient()
    print(client.invoke("你好，我是小明，我今年18岁。"))

    # test image
    print(client.invoke_with_image("你好，解释一下这个图片", "https://cloudcache.tencentcs.cn/qcloud/ui/cloud-community/build/base/images/ip-img_dd5.png"))
import os
from langchain_openai import AzureOpenAIEmbeddings

key = "bc709d6234e04a80ab2d744eb2434086"
os.environ["AZURE_OPENAI_API_KEY"] = key
api_type = "azure"
os.environ["OPENAI_API_TYPE"] = api_type
endpoint = "https://lechuang.openai.azure.com/"
os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
version = "2023-05-15"
os.environ["AZURE_OPENAI_API_VERSION"] = version
deployment = "lechuang-embedding"



def get_azure_embedding():
    """获取Azure OpenAI Embedding实例"""
    return AzureOpenAIEmbeddings(
        deployment=deployment,
        api_version=version,
        azure_endpoint=endpoint,
        api_key=key,
        model="text-embedding-ada-002"
    ) 
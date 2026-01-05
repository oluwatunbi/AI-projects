import os   
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = '2024-12-01-preview'
)

deployment = "gpt-4o"   
# craete the prompt 
prompt = "write once upon a time"
message = [{"role": "user", "content": prompt}]

completion = client.chat.completions.create(model = deployment, messages = message)

print(completion.choices[0].message.content)
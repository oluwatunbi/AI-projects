import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from PIL import Image
import requests
import json

load_dotenv()

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = '2024-12-01-preview'
)

model = 'dall-e-3'

result = client.images.generate(
    model=model,
    prompt = "A futuristic hospital in mars colony with advanced health care workers and robots",
    size = "1024x1024",
    n= 1
    
)

generate_url = json.loads(result.model_dump_json())
#
image_dir = os.path.join(os.curdir, 'images')

if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
    
image_path = os.path.join(image_dir, 'generated_image.png')

image_url = generate_url['data'][0]['url']

generate_image = requests.get(image_url).content
with open(image_path, 'wb') as image_file:
    image_file.write(generate_image)
    
image = Image.open(image_path)
image.show()
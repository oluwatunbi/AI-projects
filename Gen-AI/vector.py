import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np 

load_dotenv()

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = '2024-12-01-preview'  
    
)

# write the function for vector similarity
def counsine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# try the code
text = "the quick brown fox jumps over the lazy dog"
model = "text-embedding-ada-002"

response = client.embeddings.create(input = [text], model=model ).data[0].embedding

# texting several words
automobile_embedding = client.embeddings.create(input = 'automobile', model = model).data[0].embedding
vehicle_embedding = client.embeddings.create(input = 'vehicle', model = model).data[0].embedding
dinosaur_embedding = client.embeddings.create(input = 'dinosaur', model = model).data[0].embedding
stick_embedding = client.embeddings.create(input = 'stick', model = model).data[0].embedding

print(counsine_similarity(automobile_embedding, automobile_embedding))
print(counsine_similarity(automobile_embedding, vehicle_embedding))




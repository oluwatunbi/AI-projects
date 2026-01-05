import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# create the client
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = '2024-12-01-preview'
)

deployment = 'gpt-4o'

#create the prompt and message 
persona = input("Tell me the historic character i want to be:")
question = input("Ask your question about the hsitoric character:")

prompt= f""" You are going to play as a historic character {persona}
         whenever certain questions are asked you need to remember facts about the timeline and incidents then give accurate results only
         provide answers to the {question}
         all in just 3 lines 
         """
         
message = [{"role": "user", "content": prompt}]

completion = client.chat.completions.create(model = deployment, messages = message)

print(completion.choices[0].message.content) 

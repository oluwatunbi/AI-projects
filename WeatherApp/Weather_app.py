import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
)


# -----------------------------
# PYTHON FUNCTION TO FETCH WEATHER
# -----------------------------
def get_weather(city: str, unit: str = "celius") -> str:
    api_key = os.getenv("WEATHER_API_KEY")
    url = "http://api.weatherapi.com/v1/current.json"

    params = {"key": api_key, "q": city}

    response = requests.get(url, params=params)
    data = response.json()
    return data


# -----------------------------
# FUNCTIONS LIST FOR LLM
# -----------------------------
functions = [
    {
        "name": "get_weather",
        "description": "Get the weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get weather for"
                }
            },
            "required": ["city"]
        }
    }
]

deployment = "gpt-4o-mini"


# -----------------------------
# CHAT LOOP
# -----------------------------
is_complete = False
while not is_complete:

    userinput = input("Enter your prompt: ")

    if not userinput:
        continue

    if userinput.lower() == "exit":
        break

    messages = [{"role": "user", "content": userinput}]

    # FIRST CALL: Let LLM decide if it wants to call a function
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    message_response = completion.choices[0].message
    print("\nLLM Response:", message_response)

    # If the LLM wants to call our function
    if message_response.function_call:

        func_name = message_response.function_call.name
        function_args = json.loads(message_response.function_call.arguments)

        # Execute Python function
        result = get_weather(**function_args)

        # Add assistant function call
        messages.append({
            "role": "assistant",
            "function_call": {
                "name": func_name,
                "arguments": message_response.function_call.arguments
            }
        })

        # Add function result
        messages.append({
            "role": "function",
            "name": func_name,
            "content": json.dumps(result)
        })

        # SECOND CALL — Ask LLM to summarize the weather
        final_completion = client.chat.completions.create(
            model=deployment,
            messages=messages
        )

        print("\nFINAL SUMMARY:")
        print(final_completion.choices[0].message.content)


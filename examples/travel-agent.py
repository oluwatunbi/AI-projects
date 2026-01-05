import asyncio
import os
import requests
import logging

from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.agents import SequentialOrchestration, ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory, ChatHistoryTruncationReducer
from semantic_kernel.functions import kernel_function
from semantic_kernel.utils.logging import setup_logging
from dotenv import load_dotenv
from rich.logging import RichHandler

# Create a kernel instance 
def create_kernel() -> Kernel:
    """Creates a Kernel instance with an Azure OpenAI ChatCompletion service."""
    kernel = Kernel() 

    chat_client = AsyncOpenAI(
        api_key= os.environ["GITHUB_TOKEN"],
        base_url="https://models.inference.ai.azure.com")
    chat_completion_service = OpenAIChatCompletion(ai_model_id = "gpt-4o", async_client = chat_client)
    kernel.add_service(chat_completion_service)
    return kernel
    
async def main():
    print("Hello Travel Agent")

    # create a single kernel instance for all agents.
    kernel = create_kernel()

    # setuplogger with rich 
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    logger = logging.getLogger("rich")
    logger.info("Starting the Travel Agent...")
    
    load_dotenv()
    
    # Create a function tool with the google maps API
    Google_maps_api = os.environ["GOOGLE_MAPS_KEY"]
    
    @kernel_function(
        name = "get_vaccine_locations",
        description = " Get the locations of vaccination centres in a city and country"
    )
    
    def get_vaccine_locations(city: str, country: str):
        
        print(f"DEBUG: inside get_vaccine_locations with {city}, {country}")
        
        return [
            {"name": "Health Clinic A", "address": "123 Main St, " + city + ", " + country},
        ]
      
    # Register it as a plugin
    kernel.add_plugin(plugin=get_vaccine_locations, plugin_name="VaccineLocatorPlugin")
    
    # async def get_vaccine_locations(city: str, country: str):
    #     """ Find vacination center suing google map api"""
    #     query = f"vaccination centre in {city}, {country}"
    #     url = (
    #        f"https://maps.googleapis.com/maps/api/place/textsearch/json?"
    #        f"query={query}&key={Google_maps_api}" 
    #     )
        
    #     response = requests.get(url)
    #     data = response.json()
        
    #     result = []
        
    #     for place in data.get("results", []):
    #         result.append({
    #             "name" : place.get("name"),
    #             "address": place.get("formatted_address")
    #         })
            
    #     return result[:5]
            
        
    
 

    
    # create a diffent agent for each task
    Disease_intelligent_agent = ChatCompletionAgent(
        kernel = kernel,
        name = "Disease_intelligent_agent",
        instructions = """ You are a travel health agent that help travellers find endemic
        and outbreak prone diseases in a country or city the user is travelling to using 
        the right tools and ensuring information is accurate and up to date 
        Your priority should be 10 most important dieases 
        outline vaccines available"""
    )

    vaccine_locator_agent = ChatCompletionAgent(
        kernel = kernel,
        name = "vaccine_locator_agent",
        plugins = [get_vaccine_locations],
        
        instructions = """ 
        You are a travel health agent that help travellers find vaccination centres
        in a country or city once you get the disease infromation from the 
        {Disease_intelligent_agent} use the city or country to fine vaccination centres close 
        to the user using get_vaccine_locations function.
        """
    )

    # Create the sequential orchestration 
    sequential_orchestration = SequentialOrchestration(
        members = [Disease_intelligent_agent, vaccine_locator_agent],
 
    )
    
    # Use InProcessRuntime to run the orchestration
    runtime = InProcessRuntime()
    runtime.start()


    # create a chat loop between agent and user
    while True:
        print(" please enter the country/city you wish to visit")
        user_input = input("User: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

    # create histrory of the conversation
        history = ChatHistory(
            reducer=ChatHistoryTruncationReducer(
                max_tokens=1000,
                target_count = 1))

        history.add_user_message(user_input)

        
        orchestration_result = await sequential_orchestration.invoke(user_input, runtime = runtime)
        value = await orchestration_result.get(timeout= 60)
        print("\n *** final result ***")
        print(value)
      


        # reset conversation history
        history.clear()


if __name__ == "__main__":
    asyncio.run(main())

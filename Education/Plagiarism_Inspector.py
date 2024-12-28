import autogen
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import datetime
import os

from typing import Annotated
from tavily import TavilyClient
from pydantic import BaseModel, Field
from autogen import register_function, ConversableAgent
from dotenv import load_dotenv


OPENAI_API_KEY = os.environ.get('OPEN_AI_API_KEY')

# Configuration du modèle LLM
llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": OPENAI_API_KEY}],
    "temperature": 0
}


class TavilySearchInput(BaseModel):
    query: Annotated[str, Field(description="The search query string")]
    max_results: Annotated[
        int, Field(description="Maximum number of results to return", ge=1, le=10)
    ] = 5
    search_depth: Annotated[
        str,
        Field(
            description="Search depth: 'basic' or 'advanced'",
            choices=["basic", "advanced"],
        ),
    ] = "basic"

def tavily_search(
    input: Annotated[TavilySearchInput, "Input for Tavily search"]
) -> str:
    # Initialize the Tavily client with your API key
    client = TavilyClient(api_key=os.getenv("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"))

    # Perform the search
    response = client.search(
        query=input.query,
        max_results=input.max_results,
        search_depth=input.search_depth,
    )

    # Format the results
    formatted_results = []
    for result in response.get("results", []):
        formatted_results.append(
            f"Title: {result['title']}\\nURL: {result['url']}\\nContent: {result['content']}\\n"
        )

    return "\\n".join(formatted_results)



# Création de l'agent assistant
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant with access to internet search capabilities.",
    llm_config=llm_config,
)

# Création de l'agent assistant
similarity_evaluator = AssistantAgent(
    name="similarity_evaluator",
    llm_config=llm_config,
    system_message=f"""You are a helpful AI assistant. Your task is to establish a rating for the similarity between the text submitted by the student and 
    the texts form the internet.
    You will use at least 2 clear and distinct metrics to do so.
    Provide your feedback in a clear and constructive manner."""
)


# Création de l'agent proxy utilisateur
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    }
)


register_function(
    tavily_search,
    caller=assistant,
    executor=user_proxy,
    name="tavily_search",
    description="A tool to search the internet using the Tavily API",
)


# Message initial
today = datetime.datetime.now().date()
message = f"""Here is the essay I need you to analyze. : 
 Ah, the eternal culinary conundrum: the symphony of fiber and the robust rhythm of meat. Can these titans of the nutritional world truly coexist on the daily plate, 
 or is it a gastronomic battle for dominance? Fear not, for we shall delve into this delicious debate, weaving a tapestry of flavors and facts.
Fiber, that unsung hero of the digestive system, is the verdant embrace of vegetables, fruits, and whole grains. It dances through our intestines, 
sweeping away the debris of modern diets – the insidious sugars, the lurking fats. Fiber, you see, is a veritable broom, keeping our internal machinery humming along smoothly. 
It feeds the beneficial bacteria residing within our gut, a vibrant ecosystem crucial for overall well-being. Moreover, this fibrous friend helps us feel satiated, a delightful 
side effect that can temper those pesky cravings and keep our waistlines in check.

But what of meat, that cornerstone of many diets, that rich tapestry of flavor and texture? Meat, my dear reader, is a veritable powerhouse of protein, the building blocks of
 our very being. It provides essential amino acids, the intricate puzzle pieces that our bodies use to construct and repair tissues, from the delicate muscles that allow us to
   dance to the robust strength that allows us to carry heavy loads. Iron, that vital mineral for oxygen transport, abounds in meat, especially in its red varieties. And let us
     not forget the symphony of vitamins and minerals that accompany this culinary delight, a veritable orchestra of nutrients playing a vital role in countless bodily functions.

Now, the question arises: can these two titans, fiber and meat, truly coexist on the daily plate? The answer, my friends, is a resounding yes! A well-balanced diet, a culinary
 masterpiece, embraces the diversity of the food kingdom. Imagine a vibrant salad, a verdant canvas adorned with the juicy bursts of berries, the crunchy texture of nuts, and
 the lean embrace of grilled chicken or fish. This is the harmonious marriage of fiber and meat, a symphony of flavors that nourishes both body and soul. So, let us embrace the
   bounty of the earth, let us savor the textures, the aromas, the sheer joy of a well-crafted meal, and let us nourish ourselves with the wisdom of culinary harmony."""

# Initiation du chat
#user_proxy.initiate_chat(
#    assistant,
#    message=message
#)


# Création d'un GroupChat
groupchat = GroupChat(
    agents=[user_proxy, 
            assistant,
            similarity_evaluator,
             ],
    messages=[],
    max_round=12
)

# Création d'un manager pour le GroupChat
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Initiation de la conversation
user_proxy.initiate_chat(
    manager,
    message=message
)

def main():
    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break

        # Initiate a chat between the user proxy and the assistant
        chat_result = user_proxy.initiate_chat(
            manager,
            message=user_input,
            max_turns=2,
        )

        # Extract the assistant's reply from the chat history
        reply = next(
            (
                msg["content"]
                for msg in chat_result.chat_history
                if msg.get("name") == "Assistant"
            ),
            "I apologize, but I couldn't generate a response.",
        )

        print(f"Chatbot: {reply}")
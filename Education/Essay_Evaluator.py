import autogen
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.web_surfer import WebSurferAgent
import datetime
import os


OPENAI_API_KEY = os.environ.get('OPEN_AI_API_KEY')
# Configuration du modèle LLM
llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": OPENAI_API_KEY}],
    "temperature": 0
}



# Création de l'agent assistant
style_reviewer = AssistantAgent(
    name="style_reviewer",
    llm_config=llm_config,
    system_message=f"""You are a helpful AI assistant. Your task is to establish a rating for the style of the essay.
    You will use 4 clear and distinct metrics to do so.
    Provide your feedback in a clear and constructive manner."""
)

# Création de l'agent assistant pour la révision du code
grammar_reviewer = AssistantAgent(
    name="grammar_reviewer",
    llm_config=llm_config,
    system_message="""You are a helpful AI assistant. Your task is to establish a rating for the style of the essay. Focus on:
    Provide your feedback in a clear and constructive manner."""
)

# Création de l'agent générateur de graphiques
essay_rewriter = AssistantAgent(
    name="essay_rewriter",
    llm_config=llm_config,
    system_message="""You are a helpful AI assistant. Your task is to provide suggestions of modification for parts of the essay that need to be re-written. 
    You will give at least 2 propositions for improvement.
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

# Message initial
today = datetime.datetime.now().date()
message = f"""Here is the essay I need you to analyze in order to give me possible improvement thereof. : 
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
            style_reviewer,
            grammar_reviewer,
             essay_rewriter, ],
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
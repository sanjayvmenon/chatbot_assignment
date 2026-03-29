import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)
memory = ConversationBufferWindowMemory(
    k=5,
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent AI assistant.

Rules:
- Use previous conversation for context.
- Do NOT repeat answers.
- If unsure, ask clarification questions.
- Do NOT hallucinate.
"""),

    
    MessagesPlaceholder(variable_name="history"),

    ("human", "{input}")
])
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)
print("Chatbot is running! Type 'exit' to stop.\n")

while True:
    user_input = input("USER: ")

    if user_input.lower() == "exit":
        break

    response = conversation.predict(input=user_input)
    print("Bot:", response)

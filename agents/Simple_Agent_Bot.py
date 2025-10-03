from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv 

load_dotenv()

class AgentState(TypedDict):
    message: List[HumanMessage]
    
llm = ChatOpenAI(model="gpt-4o")    

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    print("Agent Response:", response.content)
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Your message is: ")
while user_input.lower() != "exit":
    agent.invoke({"message": [HumanMessage(content=user_input)]})
    user_input = input("Your message is: ")

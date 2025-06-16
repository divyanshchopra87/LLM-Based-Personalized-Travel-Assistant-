# LLM-Based-Personalized-Travel-Assistant-
!pip install langchain langchain_core langchain_groq langchain_community langchain langgraph
import os
from typing import TypedDict,Annotated,List
from langgraph.graph import StateGraph,END
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display,Image
# define agent 
class PlannerState(TypedDict):
  message : Annotated[List[HumanMessage|AIMessage],"the message in the conversation"]
  city : str
  interests : List[str]
  itinerary : str
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature=0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile"
)
# result =llm.invoke("what is Multi ai agent")
# result 
from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature=0,
    # Replace with your actual Groq API key
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile"
)
result =llm.invoke("what is Multi ai agent")
result
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system","You're a helpful travel assistant. create a day trip itinerary for {city} based on the user's interest :{interests}, provide a brief,bulleted itineray"),
    ("human","create an itinerary for my day trip")
])
# define agnet finctions
def input_city(state:PlannerState)-> PlannerState:
  print("please enter the city you want to travel:")
  user_message = input("your input")
  return{
      **state,
      "city":user_message,
      "message":state['message']+[HumanMessage(content=user_message)]
  }
def input_interest(state:PlannerState)->  PlannerState:
  print(f"Please enter your interest for the trip to : {state['city']}(comma-separated-value)")
  user_message = input("Your Input:")
  return{
      **state,
      "interests":[interest.strip() for interest in user_message.split(",")],
      "message":state['message']+[HumanMessage(content=user_message)]
  }
def create_itinerary(state:PlannerState)->  PlannerState:
  print(f"Creating an itinery for {state['city']}based on interests:{''.join(state['interests'])}"),
  response = llm.invoke(itinerary_prompt.format_prompt(city = state['city'],interests=','.join(state['interests'])))
  print("\nFinal Itinerary:")
  print(response.content)
  return{
      **state,
      "messages":state['message']+[AIMessage(content=response.content)],
      "itinerary":response.content
  }
# Create and compile the graph
workflow = StateGraph(PlannerState)
workflow.add_node("input_city",input_city)
workflow.add_node("input_interest",input_interest)
workflow.add_node("create_itinerary",create_itinerary)
workflow.set_entry_point("input_city")
workflow.add_edge("input_city","input_interest")
workflow.add_edge("input_interest","create_itinerary")
workflow.add_edge("create_itinerary",END)
app = workflow.compile()
display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method= MermaidDrawMethod.API
        )
    )
)
# define the function that runs the graph
def travel_planner(user_request: str):
  print("initial Request {user_request}\n")
  state = {
      "message":[HumanMessage(content=user_request)],
      "city":"",
      "interests":[],
      "itinerary":"",
  }
  for output in app.stream(state):
    pass
 user_request = "i want to plan a day trip"
travel_planner(user_request)
# defining the web interface 
# install gradio 
!pip install gradio
# setting up the same code for web site
import gradio as gr
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip."),
])

def input_city(city: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "city": city,
        "messages": state['messages'] + [HumanMessage(content=city)],
    }

def input_interests(interests: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "interests": [interest.strip() for interest in interests.split(',')],
        "messages": state['messages'] + [HumanMessage(content=interests)],
    }

def create_itinerary(state: PlannerState) -> str:
    response = llm.invoke(itinerary_prompt.format_messages(city=state['city'], interests=", ".join(state['interests'])))
    state["itinerary"] = response.content
    state["messages"] += [AIMessage(content=response.content)]
    return response.content

def travel_planner(city: str, interests: str):
    state = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
    }
    state = input_city(city, state)
    state = input_interests(interests, state)
    itinerary = create_itinerary(state)
    return itinerary

interface = gr.Interface(
    fn=travel_planner,
    theme='Yntec/HaleyCH_Theme_Orange_Green',
    inputs=[
        gr.Textbox(label="Enter the city for your day trip"),
        gr.Textbox(label="Enter your interests (comma-separated)"),
    ],
    outputs=gr.Textbox(label="Generated Itinerary"),
    title="Travel Itinerary Planner",
    description="Enter a city and your interests to generate a personalized day trip itinerary."
)

interface.launch()

  
































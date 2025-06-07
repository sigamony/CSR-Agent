import sqlite3
from typing import Optional
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from langchain.tools import tool
from dotenv import load_dotenv
import os
import re
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph import START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
import chainlit as cl

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# Define structured output schema
class AppointmentRequest(BaseModel):
    patient_name: str = Field(..., description="Full name of the patient")
    doctor_last_name: str = Field(..., description="Last name of the doctor")
    appointment_date: str = Field(..., description="Date of appointment, e.g. '2025-05-12'")
    appointment_time: str = Field(..., description="Start time of the appointment, e.g. '15:00'")
    appointment_type: Optional[str] = Field("Consultation", description="Type of appointment")

parser = PydanticOutputParser(pydantic_object=AppointmentRequest)

prompt = PromptTemplate(
    template="""
Extract structured appointment data from this request:

{request}

{format_instructions}
""",
    input_variables=["request"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API_KEY  # Pass the API key here!
)

chain: Runnable = prompt | model | parser

@tool
def add_appointment(natural_language_request: str) -> str:
    """
    Parse a natural language appointment request and add it to the database.
    """
    import traceback
    try:
        # Create a new connection and cursor for this thread
        conn = sqlite3.connect("clinic.db")
        cursor = conn.cursor()

        structured = chain.invoke({"request": natural_language_request})
        print("Structured output:", structured)
        patient_name = structured.patient_name.strip()
        doctor_last_name = structured.doctor_last_name.strip()
        date = structured.appointment_date.strip()
        time = structured.appointment_time.strip()
        appt_type = structured.appointment_type.strip().capitalize()
        print(f"Parsed values: patient_name={patient_name}, doctor_last_name={doctor_last_name}, date={date}, time={time}, appt_type={appt_type}")

        # Get PatientID
        cursor.execute("SELECT PatientID FROM Patients WHERE FirstName || ' ' || LastName = ?", (patient_name,))
        patient = cursor.fetchone()
        print("Patient query result:", patient)
        if not patient:
            conn.close()
            return f"❌ Patient '{patient_name}' not found."
        patient_id = patient[0]

        # Get DoctorID
        cursor.execute("SELECT DoctorID FROM Doctors WHERE LastName = ?", (doctor_last_name,))
        doctor = cursor.fetchone()
        print("Doctor query result:", doctor)
        if not doctor:
            conn.close()
            return f"❌ Doctor '{doctor_last_name}' not found."
        doctor_id = doctor[0]

        # Dummy Agent
        agent_id = 1

        # Insert Appointment
        cursor.execute("""
            INSERT INTO Appointments (
                PatientID, DoctorID, AppointmentDate, StartTime, EndTime, Status,
                AppointmentType, Notes, CreatedBy, CreatedAt
            ) VALUES (?, ?, ?, ?, ?, 'Scheduled', ?, '', ?, datetime('now'))
        """, (patient_id, doctor_id, date, time, time, appt_type, agent_id))

        conn.commit()
        conn.close()

        return f"✅ Appointment booked for {patient_name} with Dr. {doctor_last_name} on {date} at {time}."

    except Exception as e:
        print("Exception occurred:", e)
        print(traceback.format_exc())
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}"

# LangGraph State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]

# LLM Node definition
class LLMNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: State):
        return {"messages": [self.llm.invoke(state["messages"])]}

# Tool setup
from langchain.tools import Tool
tools = [
    Tool.from_function(
        add_appointment,
        name="add_appointment",
        description="Parse a natural language appointment request and add it to the database."
    )
]

llm_node = LLMNode(model.bind_tools(tools))
tool_node = ToolNode(tools)

# LangGraph definition
graph_builder = StateGraph(State)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", tools_condition)
graph_builder.add_edge("tools", "llm")

checkpointer = InMemorySaver()
agent = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "appointment_agent2"}}

# Async handler for external use
async def handle_appointment_message(message_content: str) -> str:
    user_msg = HumanMessage(content=message_content)
    events = agent.stream({"messages": [user_msg]}, config=config)
    full_response = ""
    for event in events:
        for value in event.values():
            msg = value["messages"][-1]
            if isinstance(msg, AIMessage):
                full_response += msg.content
    return full_response

# 📲 Chainlit message handler
@cl.on_message

async def chainlit_handle_message(message: cl.Message):


    response = await handle_appointment_message(message.content)
    await cl.Message(content=response).send()

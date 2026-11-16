from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

# ---------------------------------------------------------
# 1. PYDANTIC SCHEMA: Define the output shape we want
# ---------------------------------------------------------
class TicketExtraction(BaseModel):
    customer_email: str = Field(description="The sender's email address")
    category: Literal["billing", "bug", "feature_request"] = Field(
        description="The department best suited to handle this issue"
    )
    urgency: int = Field(
        ge=1, le=5, description="1 = low priority, 5 = production outage / double charge"
    )
    summary: str = Field(description="A crisp one-sentence summary of the issue")

# ---------------------------------------------------------
# 2. LANGCHAIN TOOL: A real Python function the LLM can call
# ---------------------------------------------------------
@tool
def escalate_to_oncall(customer_email: str, reason: str) -> str:
    """Escalate high-urgency (4 or 5) billing or outage tickets to the on-call engineer."""
    # In a real backend, this would hit PagerDuty or a Slack webhook
    print(f"\n[TOOL EXECUTED] Paging on-call for {customer_email} | Reason: {reason}")
    return f"Incident ticket opened for {customer_email}."

# ---------------------------------------------------------
# 3. INITIALIZE MODEL & RUN THE PIPELINE
# ---------------------------------------------------------
# Using init_chat_model makes swapping providers a 1-line string change
llm = init_chat_model("google_genai:gemini-3.5-flash-lite", temperature=0)

raw_email = """
From: jane.doe@acme.com
Subject: URGENT - Charged twice!!
Hey team, I just checked our corporate card and invoice #4821 was billed twice 
for $499. Can someone fix this immediately before our finance close today?
"""

# --- PHASE A: Structured Extraction ---
# Wraps the LLM so it MUST return a validated TicketExtraction instance
structured_llm = llm.with_structured_output(TicketExtraction)
ticket: TicketExtraction = structured_llm.invoke(
    f"Extract ticket details from this email:\n\n{raw_email}"
)

# --- PHASE B: Tool Decision & Execution ---
# bind_tools() attaches the JSON schema of our function to the LLM request.
# Note: The LLM does NOT run the Python function itself—it returns a tool_call request!
llm_with_tools = llm.bind_tools([escalate_to_oncall])

decision = llm_with_tools.invoke(
    f"Review this extracted ticket. If urgency >= 4, escalate it using your tool:\n"
    f"{ticket.model_dump_json()}"
)

print("\n--- 2. Model Tool Decision ---")
if decision.tool_calls:
    for call in decision.tool_calls:
        print(f"Model requested tool: '{call['name']}' with args: {call['args']}")
        # In pure LangChain (before LangGraph), YOU execute the function with those args:
        tool_result = escalate_to_oncall.invoke(call["args"])
        print(f"Tool output: {tool_result}")
else:
    print("No escalation needed:", decision.content)
"""
Natural Disaster & Weather Chatbot
Powered by OpenAI (function calling) + EM-DAT MCP server (1900-2021)
"""

import json
from typing import List, Dict, Any

import requests
import streamlit as st
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

MCP_URL = "http://disaster-mcp.onrender.com/mcp"
MODEL   = "gpt-4o-mini"

SYSTEM_PROMPT = """\
You are a specialized assistant for natural disasters and extreme weather events.
You have access to the EM-DAT global disaster database covering 1900 to 2021.
The database includes floods, earthquakes, storms, droughts, epidemics,
volcanic activity, wildfires, landslides, extreme temperatures, and more.

STRICT RULE: If the user asks about anything NOT related to weather, natural
disasters, climate events, or disaster statistics, respond with this exact
sentence and nothing else:
"I talk only about natural disasters between years 1900-2021"

For disaster-related questions, use the available tools to fetch real data.
Always cite specific numbers and facts from the database in your answer.
"""

# ── MCP tools mapped to OpenAI function-calling format ───────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_disasters",
            "description": (
                "Query EM-DAT disaster events with optional filters. "
                "Returns paginated records. Use limit/offset to page through large results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country":           {"type": "string",  "description": "Country name, partial match (e.g. 'India')"},
                    "disaster_type":     {"type": "string",  "description": "Flood, Earthquake, Storm, Drought, Epidemic, Wildfire, etc."},
                    "year_from":         {"type": "integer", "description": "Start year (1900–2021)"},
                    "year_to":           {"type": "integer", "description": "End year (1900–2021)"},
                    "continent":         {"type": "string",  "description": "Africa, Americas, Asia, Europe, Oceania"},
                    "disaster_subgroup": {"type": "string",  "description": "Hydrological, Geophysical, Meteorological, Climatological, Biological"},
                    "limit":             {"type": "integer", "description": "Max records to return (default 20, max 200)"},
                    "offset":            {"type": "integer", "description": "Pagination offset (default 0)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statistics",
            "description": (
                "Aggregated statistics for a filtered subset: total events, deaths, "
                "affected people, and economic damages — broken down by type, continent, and year."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country":       {"type": "string"},
                    "disaster_type": {"type": "string"},
                    "year_from":     {"type": "integer"},
                    "year_to":       {"type": "integer"},
                    "continent":     {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_disasters",
            "description": "Top N worst disaster events ranked by deaths, number of people affected, or economic damages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric":        {"type": "string",  "enum": ["deaths", "affected", "damages"], "description": "Ranking metric"},
                    "n":             {"type": "integer", "description": "Number of results (default 10, max 100)"},
                    "country":       {"type": "string"},
                    "disaster_type": {"type": "string"},
                    "year_from":     {"type": "integer"},
                    "year_to":       {"type": "integer"},
                    "continent":     {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_filter_options",
            "description": (
                "Returns all valid filter values: list of countries, disaster types, "
                "subgroups, continents, and the dataset's year range. "
                "Call this first to discover what data is available."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_country_summary",
            "description": "Full statistical profile for a country: total events, deaths, affected, damages, and year-by-year breakdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "Country name (partial match)"},
                },
                "required": ["country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": "Free-text search for events by name, location, or country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string", "description": "Search keyword"},
                    "limit":  {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]


# ── MCP HTTP client ───────────────────────────────────────────────────────────

def call_mcp(tool_name: str, arguments: dict) -> str:
    """Call a tool on the disaster MCP server and return its text result."""
    try:
        resp = requests.post(
            MCP_URL,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return f"Tool error: {data['error'].get('message', data['error'])}"
        return data["result"]["content"][0]["text"]
    except requests.Timeout:
        return (
            "The disaster database server is waking up (cold start on free hosting). "
            "Please wait a moment and try again."
        )
    except Exception as exc:
        return f"MCP error: {exc}"


# ── OpenAI + tool-calling loop ────────────────────────────────────────────────

def ask(question: str, history: List[Dict[str, Any]]) -> str:
    """
    Send `question` to OpenAI with the conversation `history` as context.
    Executes tool calls against the MCP server until a final answer is produced.
    """
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": question},
    ]

    for _ in range(10):  # safety cap on tool-call rounds
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            # Append the assistant's tool-call request to the message thread
            messages.append({
                "role": "assistant",
                "content": msg.content,  # may be None
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
            # Execute every requested tool and feed results back
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = call_mcp(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            return msg.content or "Sorry, I could not generate a response. Please try again."

    return "Reached the tool-call limit. Please try a more specific question."


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Disaster Chatbot",
    page_icon="🌪️",
    layout="centered",
)

st.title("🌪️ Natural Disaster Chatbot")
st.caption("Powered by OpenAI · EM-DAT database · 1900–2021 · 16,000+ events")

# Validate secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error(
        "OpenAI API key not configured. "
        "Add `OPENAI_API_KEY = '...'` to `.streamlit/secrets.toml` "
        "or the Streamlit Cloud secrets panel."
    )
    st.stop()

# Session state
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []

# Sidebar
with st.sidebar:
    st.subheader("ℹ️ About")
    st.markdown(
        "Ask me anything about **natural disasters** and **extreme weather** events "
        "from the EM-DAT global database (1900–2021).\n\n"
        "**Covered events:**\n"
        "Floods · Earthquakes · Storms · Droughts · "
        "Epidemics · Volcanic activity · Wildfires · Landslides · Extreme temperatures"
    )
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Render conversation history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input form with text area (clears after submit)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your question:",
        placeholder="e.g. What were the deadliest earthquakes in Asia since 2000?",
        height=100,
    )
    submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and user_input.strip():
    question = user_input.strip()

    # Snapshot history before appending current question (used as context)
    context = list(st.session_state.history)
    st.session_state.history.append({"role": "user", "content": question})

    with st.spinner("Searching disaster records…"):
        try:
            answer = ask(question, context)
        except Exception as exc:
            answer = f"⚠️ Unexpected error: {exc}"

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()

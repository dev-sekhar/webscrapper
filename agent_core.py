# agent_core.py

import os
from typing import Literal

from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.memory import ConversationBufferMemory

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq

from tools.tool_loader import load_all_tools

# ... (MODEL_CONFIGS dictionary remains the same) ...

# Define model configurations in one place
MODEL_CONFIGS = {
    "huggingface": {
        "class": ChatHuggingFace,
        "init_args": {
            "llm": HuggingFaceEndpoint(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                task="conversational",
                huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
                temperature=0.1,
                max_new_tokens=1024,
            )
        }
    },
    "groq": {
        "class": ChatGroq,
        "init_args": {
            "model_name": "llama3-8b-8192",
            "groq_api_key": os.environ.get("GROQ_API_KEY"),
            "temperature": 0.1,
        }
    }
}


def create_agent_executor(backend: str) -> AgentExecutor:
    """
    A unified factory function to create an agent executor for a specific backend.
    """
    print(f"🛠️  Building agent with '{backend}' backend...")

    if backend not in MODEL_CONFIGS:
        raise ValueError(f"Unknown backend: {backend}. Supported backends are: {list(MODEL_CONFIGS.keys())}")

    tools = load_all_tools()
    config = MODEL_CONFIGS[backend]
    chat_model_class = config["class"]
    init_args = config["init_args"]
    chat_model = chat_model_class(**init_args)

    # A more forceful prompt that encourages a two-step process: Find, then Fetch.
    prompt_template_text = """
You are a powerful web research assistant. Your goal is to answer the user's request accurately and with detail.

**Your process MUST follow these two distinct steps:**

**Step 1: Find Relevant Sources**
- Use a search tool (`tavily_search_engine` or `google_serp_search`) to find a list of relevant URLs.
- Analyze the search results to identify the most promising URLs.

**Step 2: Fetch and Extract Detailed Information**
- Once you have identified one or more good URLs from your search, you MUST use a browsing tool (`fetch_static_content` or `browse_dynamic_page`) to visit those URLs.
- Read the full content from the webpages.
- Do NOT answer based only on the search result summaries. You must visit the pages to get the real details.

**Final Answer Formulation:**
- After visiting the pages, synthesize all the information you have gathered into a comprehensive final answer.
- Cite your sources by including the URLs you visited.
- If you cannot find the information after searching and browsing, state that clearly.

**Example Thought Process:**
Thought: The user wants the latest news about Company X. First, I need to find news articles. I will use a search tool.
Action: tavily_search_engine
Action Input: "latest news Company X"
Observation: [List of search results with URLs]
Thought: The search results provided several promising URLs. The first one, 'https://www.news.com/article1', looks most relevant. I must now visit this URL to get the full story.
Action: fetch_static_content
Action Input: "https://www.news.com/article1"
Observation: [Full text content of the article]
Thought: I have read the full article. It contains the details I need to answer the user's question. I will now formulate my final answer.
Final Answer: According to an article from news.com, the latest news about Company X is... (Source: https://www.news.com/article1)

You have access to the following tools:
{tools}

Use the following format for your thought process:

Question: the input question you must answer
{chat_history}
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

    prompt = hub.pull("hwchase17/react-chat").partial(
        instructions=prompt_template_text
    )

    agent = create_react_agent(chat_model, tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
    )
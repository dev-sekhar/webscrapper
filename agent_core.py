# agent_core.py

import os
from typing import Literal

from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.memory import ConversationBufferMemory

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq

from tools.tool_loader import load_all_tools

# Define the supported backends as a literal type for type hinting
Backend = Literal["huggingface", "groq"]

# Define model configurations in one place
MODEL_CONFIGS = {
    "huggingface": {
        "class": ChatHuggingFace,
        "init_args": {
            "llm": HuggingFaceEndpoint(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                task="conversational",
                huggingfacehub_api_token=os.environ.get(
                    "HUGGINGFACEHUB_API_TOKEN"),
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


def create_agent_executor(backend: Backend) -> AgentExecutor:
    """
    A unified factory function to create an agent executor for a specific backend.

    Args:
        backend: The desired LLM backend, either 'huggingface' or 'groq'.

    Returns:
        An configured instance of AgentExecutor.
    """
    print(f"🛠️  Building agent with '{backend}' backend...")

    if backend not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends are: {list(MODEL_CONFIGS.keys())}")

    # 1. Load the tools
    tools = load_all_tools()

    # 2. Select and instantiate the chat model based on the backend
    config = MODEL_CONFIGS[backend]
    chat_model_class = config["class"]
    init_args = config["init_args"]
    chat_model = chat_model_class(**init_args)

    # 3. Define the new, more sophisticated system prompt
    # This guides the agent to be more critical and structured in its reasoning.
    prompt_template_text = """
You are a powerful web scraping and research assistant.
Your goal is to answer the user's request accurately by navigating the web and extracting information.
Always think step-by-step.

Here is your workflow:
1.  **Analyze the Request:** Carefully examine the user's request. Identify key entities, dates, and the core question. Be critical. For example, if a user asks for information from a future date, recognize that it does not exist.
2.  **Initial Search:** Use a search tool to gather initial information and find relevant URLs. Choose the best search tool for the job.
3.  **Critique the Search Results:** Look at the search results. Do they *actually* answer the question, or do they just contain the keywords? Check the publication dates and sources of the search results to ensure they are relevant.
4.  **Follow-Up Actions:** If the initial search is insufficient, decide on a next step. This could be:
    -   Refining your search query.
    -   Using a different search tool for a different kind of information (e.g., videos vs. text).
    -   Using a browsing tool (`fetch_static_content` or `browse_dynamic_page`) to visit a promising URL from the search results to get more detail.
5.  **Synthesize and Answer:** Once you have gathered enough information from one or more steps, formulate a final, comprehensive answer. If you absolutely cannot find the information after trying multiple steps, state that clearly and explain why (e.g., "The financial year 2025-2026 has not occurred yet, so no news has been published within that period.").

You have access to the following tools:
{tools}

Use the following format for your thought process:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

    # 4. Pull the base prompt and inject our custom template
    # We use .partial() to insert our detailed instructions into the base prompt from the hub.
    prompt = hub.pull("hwchase17/react-chat").partial(
        instructions=prompt_template_text
    )

    # 5. Create the agent runnable
    agent = create_react_agent(chat_model, tools, prompt)

    # 6. Create the memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    # 7. Create and return the Agent Executor
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
    )

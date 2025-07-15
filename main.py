# main.py

import os
import pprint
from dotenv import load_dotenv
from mcp_interface import AgenticScraper, ScraperInput

# Load all environment variables from the .env file
load_dotenv()


def run_scraper():
    """
    Initializes the agent based on .env configuration and then runs
    an interactive loop for user-provided tasks.
    """

    # 1. Read the TECHNICAL configuration from the .env file
    # Default to 'groq' if the variable is not set.
    SELECTED_BACKEND = os.environ.get("AGENT_BACKEND", "groq").lower()

    # API Key Validation based on the selected backend
    if SELECTED_BACKEND == "huggingface" and not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("❌ ERROR: Backend is set to 'huggingface' in .env, but HUGGINGFACEHUB_API_TOKEN is not found.")
        return
    if SELECTED_BACKEND == "groq" and not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: Backend is set to 'groq' in .env, but GROQ_API_KEY is not found.")
        return

    try:
        print(
            f"Initializing agent with '{SELECTED_BACKEND}' backend... (This might take a moment)")
        my_scraper = AgenticScraper(backend=SELECTED_BACKEND)
        print("✅ Agent ready. Type 'exit' or 'quit' to end.")
    except Exception as e:
        print(f"❌ Could not initialize scraper: {e}")
        return

    # 2. Start the USER-FACING interactive loop
    while True:
        print("\n" + "="*60)
        # Prompt the user for the scraping task at runtime
        task_description = input("Please enter your web scraping task: ")

        if task_description.lower() in ["exit", "quit"]:
            print("Exiting agent. Goodbye!")
            break

        if not task_description:
            continue

        print(f"\n▶️ Starting Agent with Task: '{task_description}'")
        print("="*60)

        task_input = ScraperInput(task=task_description)
        result = my_scraper.invoke(task_input)

        # Print the final results
        print("\n" + "="*25 + " ✅ FINAL RESULT " + "="*25)
        print(f"\n{result.answer}")

        print("\n" + "="*23 + " 🤔 AGENT'S PATH " + "="*23)
        try:
            history = result.intermediate_steps
            for msg in history:
                content_preview = (str(msg.content)[
                                   :300] + '...') if len(str(msg.content)) > 300 else str(msg.content)
                print(f"[{msg.type}]: {content_preview}")
            # Clear memory for the next task in the loop
            my_scraper.agent_executor.memory.clear()
        except Exception as e:
            print(f"Could not retrieve agent history: {e}")


if __name__ == "__main__":
    # The entrypoint is now simple and clean
    run_scraper()

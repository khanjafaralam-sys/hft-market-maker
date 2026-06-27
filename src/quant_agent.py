from crewai import Agent, Task, Crew, LLM

# Modern CrewAI syntax for local Ollama
local_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

# Define the AI Quant Researcher
quant_researcher = Agent(
    role="Head of Quantitative Research",
    goal="Analyze backtest logs and draft a professional strategy memo for a prop trading desk.",
    backstory="You are a veteran high-frequency trading researcher. You specialize in market microstructure, limit order book dynamics, and Avellaneda-Stoikov market making. You write concise, mathematically rigorous memos.",
    verbose=True,
    allow_delegation=False,
    llm=local_llm
)

# The data we are feeding the agent
backtest_results = """
Strategy: Volume-Clocked Order Flow Imbalance (OFI) Market Maker
Asset: BTCUSDT
Raw Tick Data Compressed: 37 Million Events -> 18,500 Volume Bars
Out-of-Sample Hit Rate: 56.20%
Initial PnL (Unoptimized): -$6,208.19 (Failure mode: Adverse Selection on trending moves)
Optimized PnL: +$9,664.89
Optimal Parameters: Risk Aversion (0.1), Spread Half (4.0), Alpha Scalar (5.0).
"""

# Define the task
write_memo = Task(
    description=f"Write a 4-paragraph formal quantitative research memo based on these results: {backtest_results}. Explain how widening the spread and heavily weighting the ML Alpha solved the adverse selection problem. Address the memo to the 'Futures First Quant Desk'.",
    expected_output="A polished, professional markdown document ready to be submitted to a trading desk.",
    agent=quant_researcher
)

# Build the crew and execute
quant_crew = Crew(
    agents=[quant_researcher],
    tasks=[write_memo]
)

print("Agent is writing the research memo...")
final_memo = quant_crew.kickoff()

# Save the output (converting the new CrewOutput object to a string)
with open("Futures_First_Quant_Memo.md", "w", encoding="utf-8") as file:
    file.write(str(final_memo))

print("Memo saved successfully to Futures_First_Quant_Memo.md!")
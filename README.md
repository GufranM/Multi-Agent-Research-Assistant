## 🤖 **Multi-Agent Research Assistant (LangGraph + Advanced AI Agents)**

This project implements an autonomous, multi-agent system designed to conduct research, analyze findings, write reports, and refine the output through a continuous critique-revision loop.

The system is orchestrated using LangGraph to manage the state and transitions between specialized agents, ensuring a robust and reliable workflow. Crucially, it relies on guaranteed structured output (using Pydantic and JSON mode) for all analysis and reporting steps, ensuring data integrity and stability.

✨ Key Features
Cyclic Workflow: Uses a state machine (LangGraph) to loop between writing and critique until the final report meets quality standards.

Specialized Agents: Four distinct agents handle the end-to-end research process:

Researcher 🔍: Gathers information using tools (Web Search, Wikipedia, Calculator).

Analyst 📊: Extracts key findings, patterns, and implications from the raw research data.

Writer ✍️: Compiles the structured analysis into a professional report.

Critic 🎯: Evaluates the report and provides structured revision instructions.

Guaranteed Structured Output: All critical steps (Research, Analysis, Report, Critique) force the LLM to return valid Pydantic models (JSON), eliminating common parsing errors and dramatically improving reliability.

Hugging Face Endpoint: Utilizes the meta-llama/Llama-3.1-8B-Instruct model via the Hugging Face Inference API for powerful, yet accessible, language generation.

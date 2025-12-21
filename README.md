# 🤖 Multi-Agent Research Assistant

[![Demo](https://img.shields.io/badge/🤗-Demo%20on%20HF%20Spaces-yellow)](https://huggingface.co/spaces/GhufranAI/Multi_Agent_Research_Assistant_with_Tavily)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()

## 🌟 Overview
This project implements a production-ready Agentic AI system featuring four specialized agents that collaborate to conduct research, analyze information, and generate high-quality reports. The system autonomously selects the appropriate tools (web search, calculator, knowledge base) based on query context, demonstrating true agentic behavior.

## ✨ Features
- 🧠 Agentic AI Architecture: Autonomous decision-making with dynamic tool selection
- 🤝 Multi-Agent Collaboration: Four specialized agents working in concert
- 🔄 Iterative Refinement: Self-improving through critic feedback loops
- 🔍 Intelligent Search: AI-optimized web search with Tavily + internal knowledge base
- 🎨 Professional UI: Clean, modern Streamlit interface with real-time visualization
- 📊 Source Attribution: Full transparency with citations and confidence scores

## 🚀 Live Demo
Try it here: [https://huggingface.co/spaces/GhufranAI/Multi_Agent_Research_Assistant_with_Tavily]

## 🛠️ Tech Stack
- **LangGraph** - Agentic workflow orchestration
- **Tavily** - AI-optimized search API
- **Llama 3.1 8B** - Language model
- **Streamlit** - Web interface
- **Pydantic** - Data validation

## 📊 Architecture
**Agent Responsibilities**



**🔍 Researcher Agent**

- **Role**: Information gathering & tool orchestration
- **Tools**: Web search (Tavily), Calculator, Knowledge base
- **Decision Making**: Analyzes query to select optimal tool

  - "latest news" → Web search
  - "calculate 25*4" → Calculator
  - "explain AI" → Knowledge base


- **Output**: Raw information with source attribution

**📊 Analyst Agent**

- **Role**: Extract insights from research findings
- **Capabilities**: Pattern recognition, theme identification
- **Output**: Key points and implications

**✍️ Writer Agent**

- **Role**: Synthesize research into professional report
- **Format**: Executive summary + findings + implications + sources
- **Output**: Structured, citation-rich report

**🎯 Critic Agent**

- **Role**: Quality assurance & improvement trigger
- **Evaluation**: Scores report on completeness, clarity, sourcing
- **Decision**: Approve (≥8/10) or request revision

**Output**: Quality score + feedback

## 🎯 Use Cases
- Research current events
- Answer complex questions
- Perform calculations
- Generate comprehensive reports

## 📸 Screenshots
[<img width="1917" height="997" alt="Screenshot 2025-12-21 210704" src="https://github.com/user-attachments/assets/ad4e8ea1-ed65-4200-a480-907643668c08" />



<img width="1904" height="1029" alt="Screenshot 2025-12-21 213004" src="https://github.com/user-attachments/assets/e6c5f459-791b-4a01-94f0-8de0a8cd7747" />


<img width="1828" height="1029" alt="Screenshot 2025-12-21 213035" src="https://github.com/user-attachments/assets/670bddc6-c635-4cce-893b-40328eb35cf3" />
]

## 💻 Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 API Keys
- Hugging Face: https://huggingface.co/settings/tokens (FREE)
- Tavily: https://tavily.com/ (1,000 searches/month free)

## 📝 License
MIT

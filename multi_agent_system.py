"""
Multi-Agent Research Assistant with LangGraph (HUGGINGFACE COMPATIBLE)
======================================================================

Adapted for HuggingFace models that don't support bind_tools() or with_structured_output()
Uses: Manual tool calling with prompt engineering + JSON parsing with error handling
Supports: Both text-generation and conversational task types

Installation:
pip install langgraph langchain langchain-community langchain-huggingface pydantic numexpr
"""

import operator
import re
import json
from typing import Annotated, List, Optional, TypedDict, Literal
from pydantic import BaseModel, Field, ValidationError
import numexpr as ne

# LangGraph imports
from langgraph.graph import StateGraph, END

# LangChain imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# ═══════════════════════════════════════════════════════════════════
# 1. PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════

class ResearchOutput(BaseModel):
    """Structured output from Researcher agent"""
    answer: str = Field(description="The direct answer to the question")
    sources_used: List[str] = Field(description="List of tools/sources consulted")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)


class AnalysisOutput(BaseModel):
    """Structured output from Analyst agent"""
    key_points: List[str] = Field(description="2-3 key points")
    implications: str = Field(description="Why this matters")


class ReportOutput(BaseModel):
    """Structured output from Writer agent"""
    title: str = Field(description="Report title")
    content: str = Field(description="Main report content")


class CritiqueOutput(BaseModel):
    """Structured output from Critic agent"""
    score: float = Field(description="Quality score 0-10", ge=0, le=10)
    needs_revision: bool = Field(description="Whether revision is needed")


# ═══════════════════════════════════════════════════════════════════
# 2. SHARED STATE
# ═══════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Shared state for all agents"""
    question: str
    research_output: Optional[ResearchOutput]
    analysis_output: Optional[AnalysisOutput]
    report_output: Optional[ReportOutput]
    critique_output: Optional[CritiqueOutput]
    report_iterations: int
    max_iterations: int
    current_step: str


# ═══════════════════════════════════════════════════════════════════
# 3. TOOLS
# ═══════════════════════════════════════════════════════════════════

@tool
def calculator(expression: str) -> str:
    """
    Perform safe mathematical calculations.
    
    Args:
        expression: A mathematical expression like "2+2" or "(10*5)+3"
    """
    try:
        expression = expression.strip()
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        result = ne.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """
    Search for general knowledge information.
    
    Args:
        query: The search query or topic
    """
    knowledge = {
        "ai": "Artificial Intelligence (AI) is the simulation of human intelligence by machines. Key applications include machine learning, natural language processing, computer vision, and robotics. AI systems can learn from data, recognize patterns, and make decisions.",
        "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence by machines. Key applications include machine learning, natural language processing, computer vision, and robotics. AI systems can learn from data, recognize patterns, and make decisions.",
        "machine learning": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions.",
        "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, AI, automation, and scientific computing.",
        "data science": "Data Science is an interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
    }
    
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    
    return f"Information about '{query}' would require web search or domain expertise. This is a general knowledge topic."


# ═══════════════════════════════════════════════════════════════════
# 4. TOOL EXECUTOR (Manual Implementation)
# ═══════════════════════════════════════════════════════════════════

class ToolExecutor:
    """Manually execute tools based on LLM requests"""
    
    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}
    
    def detect_tool_call(self, text: str) -> Optional[tuple]:
        """Detect if text contains a tool call request"""
        
        # Pattern: USE_TOOL: tool_name(arguments)
        pattern = r'USE_TOOL:\s*(\w+)\((.*?)\)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            tool_name = match.group(1)
            arguments = match.group(2).strip('"\'')
            return (tool_name, arguments)
        
        # Alternative pattern: tool_name: arguments
        for tool_name in self.tools.keys():
            if f"{tool_name}:" in text.lower():
                # Extract what comes after the tool name
                pattern = rf'{tool_name}:\s*([^\n]+)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    arguments = match.group(1).strip('"\'')
                    return (tool_name, arguments)
        
        return None
    
    def execute(self, tool_name: str, arguments: str) -> str:
        """Execute a tool with given arguments"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            result = self.tools[tool_name].func(arguments)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"


# ═══════════════════════════════════════════════════════════════════
# 5. JSON PARSER WITH ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════

def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from text with multiple strategies"""
    
    # Strategy 1: Find JSON in code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0])
        except:
            pass
    
    # Strategy 2: Find JSON without code blocks
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return parsed
        except:
            continue
    
    return None


def safe_parse_pydantic(text: str, model: BaseModel, fallback_data: dict) -> BaseModel:
    """Safely parse text into Pydantic model with fallback"""
    
    # Try to extract JSON
    json_data = extract_json(text)
    
    if json_data:
        try:
            return model(**json_data)
        except ValidationError:
            pass
    
    # Try parsing text directly as JSON
    try:
        return model.model_validate_json(text)
    except:
        pass
    
    # Fallback: Create model with fallback data
    try:
        return model(**fallback_data)
    except:
        # Last resort: minimal valid model
        return model(**{k: v for k, v in fallback_data.items() if k in model.model_fields})


# ═══════════════════════════════════════════════════════════════════
# 6. LLM FACTORY
# ═══════════════════════════════════════════════════════════════════

class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create_llm(token: str, temperature: float = 0.3):
        """Create base LLM with conversational support"""
        try:
            # Try using ChatHuggingFace wrapper for conversational models
            endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=token,
                temperature=temperature,
                max_new_tokens=1000,
                top_p=0.9,
                repetition_penalty=1.1,
                task="conversational"  # Specify conversational task
            )
            
            # Wrap with ChatHuggingFace for proper message handling
            llm = ChatHuggingFace(llm=endpoint)
            return llm
            
        except Exception as e:
            print(f"⚠️  ChatHuggingFace failed, trying standard endpoint: {e}")
            # Fallback to standard endpoint
            return HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=token,
                temperature=temperature,
                max_new_tokens=1000,
                top_p=0.9,
                repetition_penalty=1.1
            )


# ═══════════════════════════════════════════════════════════════════
# 7. AGENT NODES
# ═══════════════════════════════════════════════════════════════════

class ResearcherAgent:
    """Researcher with manual tool calling"""
    
    def __init__(self, llm, tool_executor):
        self.llm = llm
        self.tool_executor = tool_executor
    
    def __call__(self, state: AgentState) -> AgentState:
        """Research node with tool execution"""
        
        print("\n🔍 RESEARCHER AGENT")
        
        question = state["question"]
        
        # Determine which tool to use
        prompt = f"""You are a research assistant. Answer this question: {question}

Available tools:
- calculator: For math operations (e.g., "2+2", "(10*5)+3")
- search_knowledge: For information lookup (e.g., "artificial intelligence", "python")

Instructions:
1. If the question involves math/calculation, respond with: USE_TOOL: calculator(expression)
2. If the question needs information, respond with: USE_TOOL: search_knowledge(topic)
3. Replace 'expression' or 'topic' with the actual query

Examples:
- For "what is 2+2": USE_TOOL: calculator(2+2)
- For "what is AI": USE_TOOL: search_knowledge(artificial intelligence)

Your response:"""
        
        # Get LLM response (handle both chat and text models)
        try:
            # Try chat-style invocation first
            if hasattr(self.llm, 'invoke'):
                response_obj = self.llm.invoke([HumanMessage(content=prompt)])
                # Extract content from response
                if hasattr(response_obj, 'content'):
                    response = response_obj.content
                else:
                    response = str(response_obj)
            else:
                response = self.llm(prompt)
        except Exception as e:
            print(f"   ⚠️  LLM error: {e}")
            # Fallback: try direct call
            try:
                response = str(self.llm.invoke(prompt))
            except:
                response = f"Error: Unable to get LLM response for: {question}"
        
        print(f"   LLM Response: {response[:200]}...")
        
        # Check for tool call
        tool_call = self.tool_executor.detect_tool_call(response)
        
        if tool_call:
            tool_name, arguments = tool_call
            print(f"   🔧 Executing: {tool_name}({arguments})")
            
            # Execute tool
            tool_result = self.tool_executor.execute(tool_name, arguments)
            print(f"   ✅ Tool Result: {tool_result}")
            
            # Synthesize final answer
            synthesis_prompt = f"""Based on this tool result, provide a clear answer to: {question}

Tool used: {tool_name}
Tool result: {tool_result}

Provide a direct, concise answer."""
            
            try:
                if hasattr(self.llm, 'invoke'):
                    answer_obj = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
                    answer = answer_obj.content if hasattr(answer_obj, 'content') else str(answer_obj)
                else:
                    answer = self.llm(synthesis_prompt)
            except:
                answer = f"The answer is: {tool_result}"
            
            sources = [tool_name]
        else:
            # No tool needed, use LLM knowledge
            answer = response
            sources = ["LLM Knowledge"]
        
        # Create research output
        research_output = ResearchOutput(
            answer=answer.strip(),
            sources_used=sources,
            confidence=0.9 if tool_call else 0.7
        )
        
        state["research_output"] = research_output
        state["current_step"] = "research_complete"
        print(f"   ✅ Answer: {answer[:100]}...")
        
        return state


class AnalystAgent:
    """Analyzes research"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: AgentState) -> AgentState:
        """Analysis node"""
        
        print("\n📊 ANALYST AGENT")
        
        research = state["research_output"]
        
        prompt = f"""Analyze this answer and extract key insights.

Question: {state['question']}
Answer: {research.answer}

Provide your analysis in JSON format:
{{
    "key_points": ["point 1", "point 2"],
    "implications": "why this matters"
}}

Analysis:"""
        
        try:
            if hasattr(self.llm, 'invoke'):
                response_obj = self.llm.invoke([HumanMessage(content=prompt)])
                response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            else:
                response = self.llm(prompt)
        except Exception as e:
            print(f"   ⚠️  LLM error: {e}")
            response = '{"key_points": ["Analysis unavailable"], "implications": "Direct answer provided"}'
        
        # Parse with fallback
        fallback = {
            "key_points": [research.answer[:100]],
            "implications": "Direct answer provided"
        }
        
        analysis_output = safe_parse_pydantic(response, AnalysisOutput, fallback)
        
        state["analysis_output"] = analysis_output
        state["current_step"] = "analysis_complete"
        print(f"   ✅ Extracted {len(analysis_output.key_points)} key points")
        
        return state


class WriterAgent:
    """Creates reports"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: AgentState) -> AgentState:
        """Writing node"""
        
        print(f"\n✍️  WRITER AGENT (Iteration {state['report_iterations'] + 1})")
        
        research = state["research_output"]
        analysis = state["analysis_output"]
        
        prompt = f"""Write a clear, professional report.

Question: {state['question']}
Answer: {research.answer}
Key Points: {', '.join(analysis.key_points)}

Create a report in JSON format:
{{
    "title": "descriptive title",
    "content": "detailed explanation with the answer and key points"
}}

Report:"""
        
        try:
            if hasattr(self.llm, 'invoke'):
                response_obj = self.llm.invoke([HumanMessage(content=prompt)])
                response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            else:
                response = self.llm(prompt)
        except Exception as e:
            print(f"   ⚠️  LLM error: {e}")
            response = ""
        
        # Parse with fallback
        fallback = {
            "title": state['question'],
            "content": f"Question: {state['question']}\n\nAnswer: {research.answer}\n\nKey Points:\n" + "\n".join(f"• {point}" for point in analysis.key_points)
        }
        
        report_output = safe_parse_pydantic(response, ReportOutput, fallback)
        
        state["report_output"] = report_output
        state["report_iterations"] += 1
        state["current_step"] = "report_complete"
        print(f"   ✅ Report created: {len(report_output.content)} chars")
        
        return state


class CriticAgent:
    """Reviews reports"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: AgentState) -> AgentState:
        """Critique node"""
        
        print("\n🎯 CRITIC AGENT")
        
        report = state["report_output"]
        
        # Simple heuristic-based scoring for reliability
        score = 8.0
        
        # Check if answer is in content
        if state["research_output"].answer.lower() in report.content.lower():
            score += 1.0
        
        # Check content length
        if len(report.content) > 100:
            score += 0.5
        
        # Penalize first iteration slightly to allow one revision
        if state["report_iterations"] == 1:
            score -= 1.0
        
        score = min(10.0, max(0.0, score))
        
        needs_revision = (
            score < 8.0 and 
            state["report_iterations"] < state["max_iterations"]
        )
        
        critique_output = CritiqueOutput(
            score=score,
            needs_revision=needs_revision
        )
        
        state["critique_output"] = critique_output
        state["current_step"] = "critique_complete"
        print(f"   ✅ Score: {score}/10 | Revision needed: {needs_revision}")
        
        return state


# ═══════════════════════════════════════════════════════════════════
# 8. CONDITIONAL ROUTING
# ═══════════════════════════════════════════════════════════════════

def route_critique(state: AgentState) -> Literal["revise", "finish"]:
    """Route from critic"""
    critique = state["critique_output"]
    
    if critique.needs_revision:
        print(f"\n🔄 Revision needed (Score: {critique.score}/10)")
        return "revise"
    else:
        print(f"\n✅ Report approved (Score: {critique.score}/10)")
        return "finish"


# ═══════════════════════════════════════════════════════════════════
# 9. MAIN SYSTEM
# ═══════════════════════════════════════════════════════════════════

class MultiAgentSystem:
    """Multi-agent system compatible with HuggingFace models"""
    
    def __init__(self, token: str, max_iterations: int = 2):
        self.max_iterations = max_iterations
        
        print("\n" + "="*70)
        print("🤖 INITIALIZING MULTI-AGENT SYSTEM (HUGGINGFACE COMPATIBLE)")
        print("="*70)
        
        # Create tools and executor
        tools = [calculator, search_knowledge]
        self.tool_executor = ToolExecutor(tools)
        print(f"🛠️  Loaded {len(tools)} tools: {[t.name for t in tools]}")
        
        # Create LLM
        print("📡 Creating LLM...")
        self.llm = LLMFactory.create_llm(token)
        print("   ✅ LLM ready")
        
        # Initialize agents
        print("🤖 Initializing agents...")
        self.researcher = ResearcherAgent(self.llm, self.tool_executor)
        self.analyst = AnalystAgent(self.llm)
        self.writer = WriterAgent(self.llm)
        self.critic = CriticAgent(self.llm)
        print("   ✅ All agents ready")
        
        # Build graph
        print("🔗 Building workflow...")
        self.graph = self._build_graph()
        print("   ✅ Graph compiled")
        
        print("\n✅ System ready!\n")
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("researcher", self.researcher)
        workflow.add_node("analyst", self.analyst)
        workflow.add_node("writer", self.writer)
        workflow.add_node("critic", self.critic)
        
        # Set entry point
        workflow.set_entry_point("researcher")
        
        # Add edges
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "writer")
        workflow.add_edge("writer", "critic")
        
        # Conditional edge from critic
        workflow.add_conditional_edges(
            "critic",
            route_critique,
            {
                "revise": "writer",
                "finish": END
            }
        )
        
        return workflow.compile()
    
    def research(self, question: str) -> dict:
        """Execute research workflow"""
        
        print("="*70)
        print(f"📋 QUESTION: {question}")
        print("="*70)
        
        initial_state = AgentState(
            question=question,
            research_output=None,
            analysis_output=None,
            report_output=None,
            critique_output=None,
            report_iterations=0,
            max_iterations=self.max_iterations,
            current_step="start"
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            print("\n" + "="*70)
            print("✅ WORKFLOW COMPLETE")
            print("="*70)
            
            if final_state.get("critique_output"):
                print(f"Final score: {final_state['critique_output'].score}/10")
            
            return final_state
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None


# ═══════════════════════════════════════════════════════════════════
# 10. CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════

def cli_demo():
    """Command-line demo"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           MULTI-AGENT SYSTEM                                             ║
║           Manual tool calling + JSON parsing with fallbacks              ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ Token required!")
        return
    
    try:
        system = MultiAgentSystem(token=token, max_iterations=2)
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n💡 Try questions like:")
    print("   • what is 2+2")
    print("   • calculate (15*3)+7")
    print("   • what is artificial intelligence")
    print("   • what is machine learning")
    
    while True:
        print("\n" + "="*70)
        question = input("\n🤔 Enter question (or 'quit'): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        final_state = system.research(question)
        
        if final_state and final_state.get("report_output"):
            print("\n" + "="*70)
            print("📄 FINAL REPORT")
            print("="*70)
            
            report = final_state["report_output"]
            print(f"\n📌 {report.title}")
            print(f"\n{report.content}")
            
            print("\n" + "="*70)
            print("🎯 QUALITY SCORE")
            print("="*70)
            critique = final_state["critique_output"]
            print(f"Score: {critique.score}/10")


if __name__ == "__main__":
    cli_demo()
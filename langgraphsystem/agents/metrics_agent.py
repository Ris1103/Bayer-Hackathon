import os
import json
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document


# --- v1/v2 Summarization (exact replacement) ---
def summarize_telemetry(raw_data: List[Dict], model: ChatOpenAI) -> str:
    """
    v1/v2 summarization - identical behavior to map_reduce.
    """
    if not raw_data:
        return "No telemetry data available."

    # 1. Documents (modern import, same API)
    docs_content = []
    for entry in raw_data:
        content = (
            f"Time: {entry.get('timestamp')} | "
            f"Alert: {entry.get('alert', {}).get('Trigger')} | "
            f"Metrics: {entry.get('metrics', {}).get('MetricData')} | "
            f"Log: {entry.get('log_record', {}).get('message')}"
        )
        docs_content.append(content)

    # 2. Modern LCEL (replaces load_summarize_chain map_reduce)
    combine_prompt_template = """
    Summarize the following SRE telemetry logs.
    Focus on:
    1. The timeline of any latency spikes or memory increases.
    2. Any 'FATAL' or 'ERROR' messages.
    3. The frequency of ALARM vs OK statuses.
    TELEMETRY DATA:
    {text}
    CONCISE SRE SUMMARY:"""

    PROMPT = ChatPromptTemplate.from_template(combine_prompt_template)
    chain = PROMPT | model | StrOutputParser()

    return chain.invoke({"text": "\n".join(docs_content)[:8000]})  # Truncate + stuff


class MetricsAgent:
    def __init__(self, model):
        self.model = model
        self.parser = JsonOutputParser()

        self.system_prompt = """
        You are the 'Metrics Agent' for Bayer's Autonomous Incident Commander.
        Expertise: SRE, CloudWatch Telemetry, and Anomaly Detection.

        TASK:
        Analyze the provided Technical Summary of system telemetry to identify failures.
        ROOT CAUSE CATEGORIES:
        - Memory Leak: Increasing memory trend nearing 100%.
        - Latency Spike: Duration values exceeding 2000ms.
        - OOM (Out of Memory): Fatal exit errors.

        DECISION LOGIC:
        - If the summary indicates a spike but the cause is unclear, set 'needs_logs' to true.
        - If the summary explicitly mentions a 'FATAL' error or 'OOM', set 'needs_logs' to false.

        OUTPUT FORMAT (Strict JSON):
        {{
            "analysis_report": "Summary of the incident timeline and impact.",
            "root_cause": "The suspected primary failure point.",
            "needs_logs": true/false,
            "proposed_remediation": "The fix (e.g., Increase Lambda memory, Scale up, or Rollback).",
            "confidence_score": 0-100
        }}
        """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[Metrics Agent] Summarizing and analyzing telemetry...")

        raw_data = state.get("raw_data", [])
        incident_trigger = state.get("incident_trigger", "Unspecified Alert")
        telemetry_summary = summarize_telemetry(raw_data, self.model)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Trigger: {trigger}\n\nTechnical Summary of Telemetry:\n{summary}"),
            ]
        )
        chain = prompt | self.model | self.parser
        result = chain.invoke({"trigger": incident_trigger, "summary": telemetry_summary})

        next_step = "logs_agent" if result["needs_logs"] else "decision_agent"
        report_summary = (
            f"ANALYSIS: {result['analysis_report']}\n"
            f"SUSPECTED ROOT CAUSE: {result['root_cause']}"
        )

        return {
            "metrics_report": report_summary,
            "proposed_solution": result["proposed_remediation"],
            "messages": [AIMessage(content=report_summary)],
            "next_step": next_step,
        }


def metrics_node(state: Dict[str, Any]):
    if "llm" not in state or state["llm"] is None:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        llm = state["llm"]

    agent = MetricsAgent(model=llm)
    return agent.run(state)

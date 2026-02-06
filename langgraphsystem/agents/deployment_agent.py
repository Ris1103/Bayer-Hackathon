import os
import json
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    JsonOutputParser,
    StrOutputParser,
)  # Added StrOutputParser
from langchain_core.documents import Document


# --- Updated summarizer (direct replacement) ---
def summarize_deployments(deployment_data: List[Dict], model: ChatOpenAI) -> str:
    """
    Direct replacement for load_summarize_chain using v1 components.
    """
    if not deployment_data:
        return "No recent deployment or configuration change events recorded."

    docs_content = []
    for event in deployment_data:
        content = (
            f"Event Time: {event.get('timestamp')} | "
            f"Deploy ID: {event.get('deployment_id')} | "
            f"Service: {event.get('service')} | "
            f"Type: {event.get('type')} | "
            f"Details: {event.get('metadata')}"
        )
        docs_content.append(content)

    # Modern chain: Prompt | LLM | String parser (exact same logic)
    combine_prompt_template = """
    Analyze the following CI/CD deployment history.
    Focus on:
    1. Events that occurred just before the reported system failure.
    2. Changes to Database URLs, API keys, or Memory limits.
    3. Failed or rolled-back deployments.
    DEPLOYMENT EVENTS:
    {text}
    CONCISE DEPLOYMENT HISTORY SUMMARY:"""

    PROMPT = ChatPromptTemplate.from_template(combine_prompt_template)
    chain = PROMPT | model | StrOutputParser()  # Replaces load_summarize_chain

    # Join docs as single input (map_reduce behavior)
    return chain.invoke({"text": "\n".join(docs_content)[:4000]})  # Truncate for context


# --- YOUR EXACT AGENT (UNCHANGED) ---
class DeploymentAgent:
    def __init__(self, model):
        self.model = model
        self.parser = JsonOutputParser()

        self.system_prompt = """
        You are the 'Deploy Intelligence Agent' (The Historian).
        Expertise: DevOps, CI/CD pipelines, and Latent Bug Correlation.

        CONTEXT:
        Bayer's Checkout Service is failing. You have the Analysis from the Metrics and Logs Agents.
        Your task is to find the 'Smoking Gun' in the deployment history.

        TASK:
        1. Compare the time of the Incident Spike with the Deployment Timeline.
        2. Identify if a specific 'CONFIG_CHANGE' or 'CODE_DEPLOY' caused the issue.
        3. Determine if the bug was 'Latent' (i.e., the deployment happened earlier, but the error manifested later).

        OUTPUT FORMAT (Strict JSON):
        {{
            "correlated_deployment_id": "ID of the suspect deploy",
            "correlation_summary": "Explanation of how this change relates to the error.",
            "is_config_issue": true/false,
            "recommended_action": "e.g., ROLLBACK to Version X or Revert Config Change Y",
            "confidence_score": 0-100
        }}
        """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n[Deployment Agent] Correlating incident with CI/CD history...")

        # 1. Pull data from state (UNCHANGED)
        deployment_history = state.get("deployment_history", [])
        incident_trigger = state.get("incident_trigger", "Alert")
        metrics_findings = state.get("metrics_report", "No metrics info")
        logs_findings = state.get("logs_report", "No logs info")

        # 2. Summarize (now modern, same behavior)
        deploy_summary = summarize_deployments(deployment_history, self.model)

        # 3-5. EXACT SAME reasoning chain (UNCHANGED)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    f"""
            INCIDENT: {incident_trigger}
            METRICS ANALYSIS: {metrics_findings}
            LOGS ANALYSIS: {logs_findings}
            DEPLOYMENT TIMELINE:
            {deploy_summary}
            """,
                ),
            ]
        )

        chain = prompt | self.model | self.parser
        result = chain.invoke({})

        correlation_report = (
            f"DEPLOYS: Correlated with Deployment {result['correlated_deployment_id']}\n"
            f"EVIDENCE: {result['correlation_summary']}"
        )

        return {
            "deployment_report": correlation_report,
            "final_recommendation": result["recommended_action"],
            "messages": [AIMessage(content=correlation_report)],
            "next_step": "decision_agent",
        }


def deployment_node(state: Dict[str, Any]):
    if "llm" not in state or state["llm"] is None:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        llm = state["llm"]

    agent = DeploymentAgent(model=llm)
    return agent.run(state)

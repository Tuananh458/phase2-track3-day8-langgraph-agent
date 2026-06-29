# ruff: noqa: E501
"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


class Classification(BaseModel):
    route: str = Field(description="The route category. Must be one of: 'simple', 'tool', 'missing_info', 'risky', or 'error'.")
    risk_level: str = Field(description="The risk level. Must be 'high' if the route is 'risky', and 'low' otherwise.")


class Evaluation(BaseModel):
    satisfactory: bool = Field(description="True if the tool run was successful, False if it contains errors or failures")
    explanation: str = Field(description="Brief explanation of the evaluation")


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Node Implementations ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.
    """
    query = state.get("query", "").strip()

    prompt = (
        "You are an AI support ticket routing assistant. Your job is to classify the incoming support request "
        "into one of the following five routing categories:\n\n"
        "1. 'risky': Actions that have significant side effects or modifications, such as refunds, account deletions, "
        "sending confirmation emails, subscription cancellations, or other destructive/high-stakes operations.\n"
        "2. 'tool': Information lookups or queries that require database searches or checking statuses, such as "
        "looking up order status, tracking shipments, searching product lists, or querying account information.\n"
        "3. 'missing_info': Vague, brief, or incomplete requests that lack enough context or detail to take action "
        "(e.g., 'Can you fix it?', 'Help me', 'Hello', 'It is broken' with no details).\n"
        "4. 'error': Reports of system failure, service timeouts, crash alerts, or unrecoverable technical errors "
        "(e.g., 'Timeout failure while processing request', '500 Internal Server Error').\n"
        "5. 'simple': General questions, support inquiries, or informational queries that can be answered directly "
        "without lookups, tools, or high-stakes actions (e.g., 'How do I reset my password?', 'What is your return policy?').\n\n"
        "Priority order (if a query fits multiple categories, choose the highest priority category):\n"
        "risky > tool > missing_info > error > simple\n\n"
        f"Incoming User Request: \"{query}\"\n\n"
        "Classify the request. Set the 'route' to the chosen category, and set 'risk_level' to 'high' "
        "if the route is 'risky', and 'low' otherwise."
    )

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(Classification)
        classification = structured_llm.invoke(prompt)
        route = classification.route.strip().lower()
        risk_level = classification.risk_level.strip().lower()
    except Exception:
        # Fallback heuristic if LLM call fails
        lower_query = query.lower()
        if any(w in lower_query for w in ["refund", "delete", "cancel"]):
            route = "risky"
            risk_level = "high"
        elif any(w in lower_query for w in ["lookup", "status", "order", "track"]):
            route = "tool"
            risk_level = "low"
        elif any(w in lower_query for w in ["fix", "help", "hello"]) and len(lower_query.split()) < 5:
            route = "missing_info"
            risk_level = "low"
        elif any(w in lower_query for w in ["timeout", "failure", "error", "crash"]):
            route = "error"
            risk_level = "low"
        else:
            route = "simple"
            risk_level = "low"

    valid_routes = {"simple", "tool", "missing_info", "risky", "error"}
    if route not in valid_routes:
        route = "simple"
    if route == "risky":
        risk_level = "high"
    else:
        risk_level = "low"

    return {
        "route": route,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"Route classified as: {route} (risk: {risk_level})")],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list
    """
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    if route == "error" and attempt < 2:
        result_string = f"ERROR: Service timeout on attempt {attempt}"
    else:
        result_string = f"SUCCESS: Action / lookup completed successfully on attempt {attempt}"

    return {
        "tool_results": [result_string],
        "events": [make_event("tool", "completed", f"Tool executed: {result_string}")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.
    """
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""

    if "ERROR" in latest_result:
        evaluation_result = "needs_retry"
    else:
        try:
            llm = get_llm()
            structured_llm = llm.with_structured_output(Evaluation)
            prompt = (
                f"Analyze the following tool result for the user query: '{state.get('query')}'\n"
                f"Tool result: '{latest_result}'\n\n"
                "Determine if the tool result successfully resolved the request or if it contains a failure "
                "or error requiring a retry. Set 'satisfactory' to True if successful/complete, "
                "or False if it needs a retry."
            )
            eval_res = structured_llm.invoke(prompt)
            evaluation_result = "success" if eval_res.satisfactory else "needs_retry"
        except Exception:
            evaluation_result = "success"

    return {
        "evaluation_result": evaluation_result,
        "events": [make_event("evaluate", "completed", f"Evaluation result: {evaluation_result}")],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")

    context = []
    if tool_results:
        context.append(f"Tool results: {tool_results}")
    if approval:
        context.append(f"Approval status: {approval}")

    context_str = "\n".join(context)

    prompt = (
        "You are a customer support agent. Generate a final helpful and polite response to the user's query.\n"
        "You must ground your response strictly in the provided context and action status below. "
        "Do not invent facts or make up details that are not in the context.\n\n"
        f"User Query: \"{query}\"\n"
        f"Context:\n{context_str}\n\n"
        "Response:"
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        final_answer = response.content.strip()
    except Exception:
        if tool_results:
            final_answer = f"Based on the processed lookup/action: {tool_results[-1]}"
        else:
            final_answer = "Thank you for contacting support. We have successfully processed your inquiry."

    return {
        "final_answer": final_answer,
        "events": [make_event("answer", "completed", f"Response generated: {final_answer[:40]}")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.
    """
    query = state.get("query", "")
    prompt = (
        f"The user query is: '{query}'. This query is vague or missing information to perform any action.\n"
        "Generate a clear, polite, and specific clarification question asking the user for the necessary details."
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        pending_question = response.content.strip()
    except Exception:
        pending_question = "Could you please provide more details or specify what you need help with?"

    return {
        "pending_question": pending_question,
        "final_answer": pending_question,
        "events": [make_event("clarify", "completed", f"Clarification requested: {pending_question[:40]}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.
    """
    query = state.get("query", "")
    prompt = (
        f"The user requested: '{query}'. This is classified as a risky action (e.g. refund, account deletion, etc.).\n"
        "Describe the exact proposed action and explain why it requires human verification or approval before execution."
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        proposed_action = response.content.strip()
    except Exception:
        proposed_action = f"Execute high-risk operation: '{query}'"

    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "completed", f"Prepared action: {proposed_action[:40]}")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.
    """
    if os.getenv("LANGGRAPH_INTERRUPT") == "true":
        from langgraph.types import interrupt
        decision = interrupt({
            "proposed_action": state.get("proposed_action"),
            "scenario_id": state.get("scenario_id")
        })

        if isinstance(decision, dict):
            approval = {
                "approved": decision.get("approved", False),
                "reviewer": decision.get("reviewer", "human-reviewer"),
                "comment": decision.get("comment", "")
            }
        else:
            approval = {
                "approved": bool(decision),
                "reviewer": "human-reviewer",
                "comment": str(decision)
            }
    else:
        approval = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "Auto-approved for scenario " + state.get("scenario_id", ""),
        }

    return {
        "approval": approval,
        "events": [make_event("approval", "completed", f"Approval decision: {approval.get('approved')}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.
    """
    current_attempt = state.get("attempt", 0)
    new_attempt = current_attempt + 1
    error_msg = f"Transient failure during attempt {current_attempt}"
    return {
        "attempt": new_attempt,
        "errors": [error_msg],
        "events": [make_event("retry", "completed", f"Retry registered. New attempt count: {new_attempt}")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.
    """
    query = state.get("query", "")
    errors = state.get("errors", [])
    try:
        llm = get_llm()
        prompt = (
            f"The customer requested: '{query}'. However, the internal systems failed repeatedly with "
            f"the following recorded errors: {errors}.\n"
            "Generate a professional, polite response apologizing to the customer, explaining that the "
            "request could not be completed due to system error, and stating that the ticket has been "
            "escalated to support staff."
        )
        response = llm.invoke(prompt)
        final_answer = response.content.strip()
    except Exception:
        final_answer = "We apologize, but we were unable to complete your request after multiple system attempts. This issue has been escalated."

    return {
        "final_answer": final_answer,
        "events": [make_event("dead_letter", "completed", "Max attempts exceeded. Escalating.")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }

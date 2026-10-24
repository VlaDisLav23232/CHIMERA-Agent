import time
from typing import Annotated, Any, TypedDict

from chimera_agent_baseline.output.schema import eligible_variables
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from chimera_agent.decision import decide
from chimera_agent.evidence import (CASE_MISMATCH, Checks, DecisionRecord, EvidenceItem, Timings, ToolCallRecord, Trace,
                                    build_ledger, message_text, revealed_tools, run_checks)


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    agent_calls: int
    agent_s: float
    evidence: list[EvidenceItem]
    tool_calls: list[ToolCallRecord]
    agent_message: str
    decision: DecisionRecord
    checks: Checks
    decision_attempts: int
    escape_repairs: int
    decision_s: float


def case_guard(case_id: str):
    async def guard(request, execute):
        # Tools serve every staged case; a call for another patient is refused, not executed.
        requested = request.tool_call["args"].get("case_id", case_id)
        if requested != case_id:
            return ToolMessage(content=f"{CASE_MISMATCH}: this consultation is case {case_id}, not {requested}",
                               name=request.tool_call["name"], tool_call_id=request.tool_call["id"], status="error")
        return await execute(request)
    return guard


def final_agent_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls and message.content:
            return message_text(message.content)
    return ""


def create_graph(model: BaseChatModel, tools: list[BaseTool], system_prompt: str, decision_guidance: str, case_id: str):
    bound = model.bind_tools(tools)

    async def agent(state: State) -> dict[str, Any]:
        started = time.monotonic()
        response = await bound.ainvoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response], "agent_calls": state.get("agent_calls", 0) + 1,
                "agent_s": state.get("agent_s", 0.0) + time.monotonic() - started}

    async def decide_node(state: State) -> dict[str, Any]:
        started = time.monotonic()
        prompt = message_text(state["messages"][0].content)
        items, calls = build_ledger(prompt, state["messages"])
        agent_message = final_agent_text(state["messages"])
        eligible = eligible_variables(1, revealed_tools(calls))
        record, attempts, repairs = await decide(model, items, agent_message, eligible, decision_guidance)
        return {"evidence": items, "tool_calls": calls, "agent_message": agent_message, "decision": record,
                "checks": run_checks(record, items, calls), "decision_attempts": attempts, "escape_repairs": repairs,
                "decision_s": time.monotonic() - started}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools, awrap_tool_call=case_guard(case_id)))
    builder.add_node("decide", decide_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", lambda s: "tools" if tools_condition(s) == "tools" else "decide",
                                  {"tools": "tools", "decide": "decide"})
    builder.add_edge("tools", "agent")
    builder.add_edge("decide", END)
    return builder.compile()


async def run_case(model: BaseChatModel, tools: list[BaseTool], system: str, system_prompt: str, decision_guidance: str,
                   case_id: str, context: str, max_iterations: int) -> Trace:
    started = time.monotonic()
    graph = create_graph(model, tools, system_prompt, decision_guidance, case_id)
    state = await graph.ainvoke({"messages": [HumanMessage(content=context)]}, {"recursion_limit": max_iterations})
    return Trace(case_id=case_id, system=system, evidence=state["evidence"], tool_calls=state["tool_calls"],
                 agent_message=state["agent_message"], decision=state["decision"], checks=state["checks"],
                 decision_attempts=state["decision_attempts"], escape_repairs=state["escape_repairs"],
                 llm_calls=state["agent_calls"] + state["decision_attempts"],
                 timings=Timings(agent_s=round(state["agent_s"], 3), decision_s=round(state["decision_s"], 3),
                                 total_s=round(time.monotonic() - started, 3)))

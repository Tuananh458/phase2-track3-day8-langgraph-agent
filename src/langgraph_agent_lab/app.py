# ruff: noqa
import os
import sys
import uuid
import sqlite3
from pathlib import Path
import streamlit as st

# Add src folder to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Force LANGGRAPH_INTERRUPT to true for Streamlit HITL demonstration
os.environ["LANGGRAPH_INTERRUPT"] = "true"

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import initial_state, Scenario, Route
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph.types import Command

# Page configuration
st.set_page_config(
    page_title="LangGraph Agentic Lab Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style (Premium CSS)
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
    }
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.3rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #fef9c3;
        color: #a16207;
        padding: 0.3rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.3rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .timeline-node {
        font-weight: 700;
        color: #3b82f6;
    }
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# App state management helpers
@st.cache_resource
def get_compiled_graph(checkpointer_type="sqlite"):
    if checkpointer_type == "sqlite":
        # Using a dedicated database path for streamlit to avoid lock conflict with CLI runs
        db_path = "streamlit_checkpoints.db"
        checkpointer = build_checkpointer("sqlite", db_path)
    else:
        checkpointer = build_checkpointer("memory")
    return build_graph(checkpointer)

# Sidebar configurations
with st.sidebar:
    st.image("https://raw.githubusercontent.com/langchain-ai/langgraph/main/docs/static/img/langgraph_overview.png", width=180)
    st.markdown("### ⚙️ Cấu hình Hệ thống")
    
    checkpointer_choice = st.selectbox(
        "Checkpointer Backend",
        ["sqlite", "memory"],
        help="SQLite lưu trữ liên tục trạng thái trên ổ đĩa, Memory lưu trữ trong bộ nhớ tạm."
    )
    
    st.markdown("---")
    st.markdown("### 🔑 API Keys Status")
    
    openai_key_configured = bool(os.getenv("OPENAI_API_KEY"))
    gemini_key_configured = bool(os.getenv("GEMINI_API_KEY"))
    anthropic_key_configured = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    if openai_key_configured:
        st.success("OpenAI API Key: Đã cấu hình ✅")
    else:
        st.warning("OpenAI API Key: Trống ❌")
        
    if gemini_key_configured:
        st.success("Gemini API Key: Đã cấu hình ✅")
    if anthropic_key_configured:
        st.success("Anthropic API Key: Đã cấu hình ✅")
        
    st.info("LANGGRAPH_INTERRUPT: Được bật (True) để test chế độ Duyệt thủ công (HITL) thực tế.")

# Load graph
graph = get_compiled_graph(checkpointer_choice)

# Main Title Header
st.markdown("<div class='main-title'>LangGraph Support Ticket Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Hệ thống phân luồng tác vụ và xử lý Ticket thông minh sử dụng Đồ thị Trạng thái (StateGraph)</div>", unsafe_allow_html=True)

# Load sample scenarios
scenarios_path = "data/sample/scenarios.jsonl"
preset_scenarios = []
if os.path.exists(scenarios_path):
    try:
        preset_scenarios = load_scenarios(scenarios_path)
    except Exception as e:
        st.error(f"Không thể tải các kịch bản mẫu: {e}")

# Session state initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None
if "current_state" not in st.session_state:
    st.session_state.current_state = None

# Scenario Selector
st.markdown("### 🎫 Lựa chọn Ticket để chạy thử nghiệm")

scenario_options = ["Tự nhập câu hỏi (Custom)..."] + [f"{s.id} - Ex: {s.query} ({s.expected_route.value})" for s in preset_scenarios]
selected_option = st.selectbox("Chọn kịch bản mẫu hoặc nhập tự do:", scenario_options)

query_input = ""
selected_scenario_obj = None

if selected_option != "Tự nhập câu hỏi (Custom)...":
    # Get index of selected scenario
    idx = scenario_options.index(selected_option) - 1
    selected_scenario_obj = preset_scenarios[idx]
    query_input = selected_scenario_obj.query
else:
    query_input = st.text_area("Nhập nội dung Ticket khách hàng gửi vào đây:", value="Please check my order status for order 98765")

# Action Buttons
col1, col2, col3 = st.columns([2, 2, 8])

reset_clicked = col2.button("Clear / Reset", use_container_width=True)
if reset_clicked:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.current_state = None
    st.session_state.current_scenario = None
    st.rerun()

run_clicked = col1.button("Chạy thử nghiệm (Run Agent)", type="primary", use_container_width=True)

# Config and Thread details
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Run Agent logic
if run_clicked:
    # Build initial scenario object
    if selected_scenario_obj:
        scenario = selected_scenario_obj
    else:
        # Create a mock scenario
        scenario = Scenario(
            id="custom-ticket",
            query=query_input,
            expected_route=Route.SIMPLE,
            requires_approval="refund" in query_input.lower() or "delete" in query_input.lower() or "cancel" in query_input.lower(),
            should_retry=False,
            max_attempts=3
        )
    
    st.session_state.current_scenario = scenario
    
    # Initialize state
    state = initial_state(scenario)
    
    st.info(f"Đang chạy Ticket với Thread ID: `{st.session_state.thread_id}`...")
    
    try:
        # We invoke the graph with initial state
        result = graph.invoke(state, config=config)
        st.session_state.current_state = result
    except Exception as e:
        # Check if paused on interrupt
        current_graph_state = graph.get_state(config)
        if current_graph_state.next:
            st.session_state.current_state = current_graph_state.values
            st.warning(f"Đồ thị đang tạm dừng tại bước: `{current_graph_state.next}` chờ phê duyệt.")
        else:
            st.error(f"Đã xảy ra lỗi khi chạy đồ thị: {e}")

# Fetch current state from checkpoint to verify status
current_graph_state = graph.get_state(config)
state_values = current_graph_state.values if current_graph_state.values else st.session_state.current_state

if state_values:
    # Main Dashboard layout
    left_col, right_col = st.columns([7, 5])
    
    with left_col:
        st.markdown("### 📊 Trạng thái Chạy Đồ thị")
        
        # Display Route and Risk level
        route_val = state_values.get("route", "Chưa xác định")
        risk_val = state_values.get("risk_level", "low")
        attempt_val = state_values.get("attempt", 0)
        max_attempt_val = state_values.get("max_attempts", 3)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route Phân Loại", route_val.upper())
        
        # Risk Badge
        if risk_val == "high":
            c2.markdown("**Risk Level**  \n<span class='badge-danger'>HIGH</span>", unsafe_allow_html=True)
        else:
            c2.markdown("**Risk Level**  \n<span class='badge-success'>LOW</span>", unsafe_allow_html=True)
            
        c3.metric("Số lần thử (Attempts)", f"{attempt_val} / {max_attempt_val}")
        
        # Next Node Badge
        if current_graph_state.next:
            c4.markdown(f"**Trạng thái Kế Tiếp**  \n<span class='badge-warning'>PAUSED AT {current_graph_state.next}</span>", unsafe_allow_html=True)
        else:
            c4.markdown("**Trạng thái Kế Tiếp**  \n<span class='badge-success'>COMPLETED</span>", unsafe_allow_html=True)
            
        # Display Final Answer or pending actions
        st.markdown("---")
        
        final_answer = state_values.get("final_answer")
        pending_question = state_values.get("pending_question")
        
        if final_answer:
            st.markdown("#### 🎯 Câu Trả Lời Cuối Cùng (Final Answer):")
            st.success(final_answer)
        elif pending_question:
            st.markdown("#### ❓ Đang Hỏi Lại Khách Hàng (Clarification Request):")
            st.warning(pending_question)
            
        # HITL Interruption Panel (Phê duyệt thủ công)
        if current_graph_state.next and "approval" in current_graph_state.next:
            st.markdown("<div class='card' style='border-left: 5px solid #e11d48;'>", unsafe_allow_html=True)
            st.markdown("🔴 **Hệ thống đang tạm dừng: Yêu cầu Phê duyệt hành động nhạy cảm**")
            
            proposed_action = state_values.get("proposed_action", "Thực hiện hành động nguy hiểm.")
            st.write(f"**Mô tả hành động cần phê duyệt:** {proposed_action}")
            
            reviewer_comment = st.text_input("Ý kiến kiểm duyệt (Reviewer Comment):", placeholder="Lý do duyệt hoặc từ chối...")
            
            b_col1, b_col2 = st.columns(2)
            
            approve_btn = b_col1.button("Duyệt (Approve)", type="primary", use_container_width=True)
            reject_btn = b_col2.button("Từ chối (Reject)", use_container_width=True)
            
            if approve_btn:
                decision = {
                    "approved": True,
                    "reviewer": "streamlit-human",
                    "comment": reviewer_comment
                }
                st.success("Hành động đã được duyệt! Đang gửi kết quả phê duyệt và chạy tiếp...")
                # Resume graph
                try:
                    res = graph.invoke(Command(resume=decision), config=config)
                    st.session_state.current_state = res
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi tiếp tục luồng: {e}")
                    
            if reject_btn:
                decision = {
                    "approved": False,
                    "reviewer": "streamlit-human",
                    "comment": reviewer_comment
                }
                st.warning("Hành động đã bị từ chối! Tiếp tục luồng xử lý...")
                # Resume graph
                try:
                    res = graph.invoke(Command(resume=decision), config=config)
                    st.session_state.current_state = res
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi tiếp tục luồng: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Display Tool results
        tool_res_list = state_values.get("tool_results", [])
        if tool_res_list:
            st.markdown("#### 🛠️ Kết quả chạy các Công cụ (Tool Results):")
            for tr in tool_res_list:
                if "ERROR" in tr:
                    st.error(tr)
                else:
                    st.info(tr)
                    
        # Errors log
        error_list = state_values.get("errors", [])
        if error_list:
            st.markdown("#### ⚠️ Nhật ký lỗi tạm thời (Transient Errors):")
            for err in error_list:
                st.error(err)
                
    with right_col:
        st.markdown("### 📜 Nhật Ký Sự Kiện Đồ Thị (Audit Events)")
        
        events = state_values.get("events", [])
        if events:
            # Timeline structure
            for ev in events:
                node_name = ev.get("node", "unknown").upper()
                msg = ev.get("message", "")
                st.markdown(f"📍 <span class='timeline-node'>{node_name}</span>: {msg}", unsafe_allow_html=True)
        else:
            st.write("Chưa có sự kiện nào được ghi nhận.")
            
        # State Dictionary Inspector
        st.markdown("---")
        with st.expander("🔍 Chi tiết trạng thái lưu trữ (AgentState JSON)"):
            clean_state = {k: v for k, v in state_values.items() if k not in ["messages"]}
            st.json(clean_state)
else:
    st.info("Vui lòng nhấn nút **Chạy thử nghiệm (Run Agent)** để kích hoạt luồng tác vụ và kiểm thử.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from optimizer import run_pso, GRID_W, GRID_H
from llm_layer import parse_with_ollama, explain_result_with_ollama

st.set_page_config(page_title="UAV Mission Intelligence", page_icon="🚁", layout="wide")

DEFAULT_SCENARIO = """A flood has affected a town. Deploy three drones from different launch points.
Inspect a damaged bridge at (16,11) with critical priority 5, search a school at (5,11)
with priority 5, inspect a hospital at (14,4) with priority 4, check a road blockage at
(9,7) with priority 3, search a residential block at (3,3) with priority 4, and monitor
a water-treatment plant at (17,2) with priority 2. There are buildings/restricted cells
around (8,4), (8,5), (8,6), (12,8), (13,8), (14,8), (5,9), and (6,9).
Use UAVs with different battery budgets and speeds."""

DEMO_DATA = {
    "mission_name": "Flood response – demo mission",
    "uavs": [
        {"name":"UAV-1", "start":[1,1], "battery":48, "speed":1.25, "energy_per_cell":1.0},
        {"name":"UAV-2", "start":[18,13], "battery":42, "speed":1.00, "energy_per_cell":0.9},
        {"name":"UAV-3", "start":[1,13], "battery":36, "speed":0.85, "energy_per_cell":0.8}
    ],
    "tasks": [
        {"name":"Damaged bridge", "pos":[16,11], "priority":5},
        {"name":"School search", "pos":[5,11], "priority":5},
        {"name":"Hospital inspection", "pos":[14,4], "priority":4},
        {"name":"Road blockage", "pos":[9,7], "priority":3},
        {"name":"Residential search", "pos":[3,3], "priority":4},
        {"name":"Water plant", "pos":[17,2], "priority":2}
    ],
    "obstacles":[[8,4],[8,5],[8,6],[12,8],[13,8],[14,8],[5,9],[6,9]]
}

def draw_map(scenario, result=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, GRID_W-0.5)
    ax.set_ylim(-0.5, GRID_H-0.5)
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal")
    for x,y in scenario["obstacles"]:
        ax.scatter(x,y, marker="s", s=230, color="#555555")
    for i,t in enumerate(scenario["tasks"]):
        x,y=t["pos"]
        ax.scatter(x,y, marker="*", s=210, color="#f59e0b", edgecolor="black", zorder=4)
        ax.annotate(f'T{i+1}: {t["name"]}', (x,y), xytext=(5,5), textcoords="offset points", fontsize=7)
    for i,u in enumerate(scenario["uavs"]):
        x,y=u["start"]
        ax.scatter(x,y, marker="^", s=150, label=u["name"], zorder=5)
    if result:
        for i,route in enumerate(result["routes"]):
            if route["path"]:
                xs=[p[0] for p in route["path"]]
                ys=[p[1] for p in route["path"]]
                ax.plot(xs,ys,linewidth=2.3,marker=".",markersize=3,label=f'{scenario["uavs"][i]["name"]} route')
    ax.set_title("Mission map: obstacles, task locations and planned UAV routes")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.10), ncol=3, fontsize=8)
    fig.tight_layout()
    return fig

st.title("🚁 UAV Mission Intelligence")
st.caption("LLM-assisted mission understanding • PSO-based task allocation • A* obstacle-aware routing")
st.markdown(
    "A research prototype for disaster-response missions. Describe the mission in plain English, "
    "let the LLM structure the scenario, and run a bio-inspired optimizer to obtain an allocation and routes."
)

with st.sidebar:
    st.header("Mission configuration")
    model = st.text_input("Ollama model", value="qwen2.5:3b")
    st.caption("Local LLM endpoint: http://localhost:11434")
    particles = st.slider("PSO particles", 12, 60, 36, 4)
    iterations = st.slider("PSO iterations", 20, 120, 65, 5)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=7)
    st.caption("More particles/iterations may improve search, but take longer.")

scenario_text = st.text_area("Describe your disaster-response mission", value=DEFAULT_SCENARIO, height=160)
c1,c2 = st.columns([1,1])
with c1:
    use_llm = st.checkbox("Use local LLM to interpret the mission", value=True)
with c2:
    use_demo = st.checkbox("Use built-in validated demo if LLM is unavailable", value=True)

if st.button("Understand mission and optimize", type="primary", use_container_width=True):
    scenario = None
    llm_status = ""
    if use_llm:
        try:
            with st.spinner("LLM is converting your mission brief into structured UAV, task and obstacle data..."):
                scenario = parse_with_ollama(scenario_text, model=model)
            llm_status = "Mission interpreted successfully by the local LLM."
        except Exception as e:
            if use_demo:
                scenario = DEMO_DATA
                llm_status = f"LLM unavailable or returned invalid data; loaded the built-in demo scenario instead. Detail: {e}"
            else:
                st.error(f"Could not use the LLM: {e}")
                st.stop()
    else:
        scenario = DEMO_DATA
        llm_status = "Built-in demo scenario selected."
    st.session_state["scenario"] = scenario
    st.session_state["llm_status"] = llm_status
    with st.spinner("PSO is searching candidate task assignments and task orders; A* is calculating obstacle-aware routes..."):
        result = run_pso(scenario["uavs"], scenario["tasks"], scenario["obstacles"],
                         particles=int(particles), iterations=int(iterations), seed=int(seed))
    st.session_state["result"] = result
    st.session_state["model"] = model

if "scenario" in st.session_state and "result" in st.session_state:
    scenario = st.session_state["scenario"]
    result = st.session_state["result"]
    st.success(st.session_state["llm_status"])
    st.subheader(scenario.get("mission_name","Mission overview"))
    left,right = st.columns([1.5,1])
    with left:
        st.pyplot(draw_map(scenario, result), use_container_width=True)
    with right:
        st.markdown("**Mission inputs extracted**")
        st.metric("UAVs", len(scenario["uavs"]))
        st.metric("Tasks", len(scenario["tasks"]))
        st.metric("Obstacle cells", len(scenario["obstacles"]))
        st.markdown("**UAV capability table**")
        st.dataframe(pd.DataFrame([
            {"UAV":u["name"],"Start":str(u["start"]),"Battery":u["battery"],
             "Speed":u["speed"]} for u in scenario["uavs"]
        ]), hide_index=True, use_container_width=True)

    st.subheader("Optimization results")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total travel distance", f'{result["distance"]:.1f} cells')
    m2.metric("Estimated energy", f'{result["energy"]:.1f} units')
    m3.metric("Mission completion time", f'{result["makespan"]:.1f} units')
    m4.metric("Fitness (lower is better)", f'{result["fitness"]:.1f}')
    m5,m6,m7 = st.columns(3)
    m5.metric("Workload imbalance", f'{result["imbalance"]:.2f}')
    m6.metric("Collision-risk events", result["collisions"])
    m7.metric("Battery penalty", f'{result["battery_penalty"]:.1f}')

    st.markdown("### Task allocation")
    rows=[]
    for i,ids in enumerate(result["allocation"]):
        for tid in ids:
            rows.append({
                "UAV":scenario["uavs"][i]["name"],
                "Assigned task":scenario["tasks"][tid]["name"],
                "Location":str(scenario["tasks"][tid]["pos"]),
                "Priority":scenario["tasks"][tid]["priority"]
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("### Route and per-UAV metrics")
    route_rows=[]
    for i,r in enumerate(result["routes"]):
        task_names=[scenario["tasks"][t]["name"] for t in result["allocation"][i]]
        route_rows.append({
            "UAV":scenario["uavs"][i]["name"],
            "Visit order":" → ".join(task_names) if task_names else "No tasks",
            "Distance":round(r["distance"],2),
            "Energy":round(r["energy"],2),
            "Flight time":round(r["time"],2),
            "Battery budget":scenario["uavs"][i]["battery"],
            "Battery feasible": "Yes" if r["energy"] <= scenario["uavs"][i]["battery"] and r["feasible"] else "No"
        })
    st.dataframe(pd.DataFrame(route_rows), hide_index=True, use_container_width=True)

    st.markdown("### How PSO improved the candidate solution")
    fig2,ax=plt.subplots(figsize=(9,3.2))
    ax.plot(result["history"], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.set_title("Global-best fitness across PSO iterations")
    ax.grid(alpha=0.25)
    st.pyplot(fig2, use_container_width=True)

    st.markdown("### LLM-generated explanation")
    if st.button("Explain these results using the LLM"):
        try:
            with st.spinner("LLM is explaining the computed results..."):
                explanation=explain_result_with_ollama(scenario,result,model=st.session_state["model"])
            st.session_state["explanation"]=explanation
        except Exception as e:
            st.warning(f"Could not reach Ollama for explanation: {e}")
    if "explanation" in st.session_state:
        st.write(st.session_state["explanation"])

    with st.expander("View the numerical objective components"):
        st.write({
            "Distance contribution": round(result["distance"],2),
            "Energy contribution (weight 1.5)": round(1.5*result["energy"],2),
            "Completion-time contribution (weight 3)": round(3*result["makespan"],2),
            "Workload-imbalance contribution (weight 2)": round(2*result["imbalance"],2),
            "Priority-delay contribution (weight 0.8)": round(0.8*result["priority_delay"],2),
            "Collision penalty (120 per estimated event)": round(120*result["collisions"],2),
            "Battery penalty": round(result["battery_penalty"],2)
        })

st.divider()
st.caption(
    "Academic prototype only: 2D grid simulation with simplified energy and time models. "
    "It is not connected to real drones, GPS, live weather or certified collision-avoidance systems. "
    "The PSO weights are demonstration parameters and should be reported as such."
)

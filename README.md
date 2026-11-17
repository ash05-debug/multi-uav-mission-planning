# UAV Mission Intelligence – LLM + PSO research prototype

A moderate, local-hosted demonstration aligned with the research assignment
**“Bio-Inspired Multi-UAV Task Allocation and Path Planning.”**

## What it demonstrates

1. Natural-language disaster mission brief.
2. Local LLM (Ollama) extracts structured UAV, task, priority and obstacle data.
3. Discrete Particle Swarm Optimization (PSO) searches for task assignments and visit orders.
4. A* finds obstacle-avoiding grid paths for each UAV.
5. A multi-objective fitness function scores distance, energy, mission time, workload imbalance,
   priority delay, collision-risk estimates and battery-budget violations.
6. Streamlit dashboard displays the map, assignment, routes, metrics, convergence curve and
   an LLM explanation of the computed result.

## Setup (Windows / VS Code)

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install packages

```powershell
pip install -r requirements.txt
```

### 3. Install and start Ollama (recommended for the LLM layer)

Install Ollama from https://ollama.com/download, then in a terminal run:

```powershell
ollama pull qwen2.5:3b
ollama serve
```

If Ollama is already running as a background service, you may only need the `ollama pull` command.
The app expects the default endpoint `http://localhost:11434`.

### 4. Run the application

Open another terminal in this project folder:

```powershell
streamlit run app.py
```

The local dashboard normally opens at `http://localhost:8501`.

## Demo flow for faculty

1. Open the app and show the natural-language flood-response brief.
2. Keep “Use local LLM to interpret the mission” enabled.
3. Click **Understand mission and optimize**.
4. Explain the extracted mission entities: UAV start points, battery, speed, task coordinates,
   task priorities and obstacles.
5. Show the planned routes and task allocation.
6. Show the metric cards and explain that the fitness is a weighted objective (lower is better).
7. Show the PSO convergence graph: it records the best fitness found after each iteration.
8. Click **Explain these results using the LLM** to get a plain-language explanation grounded in
   the actual computed metrics.

If Ollama is unavailable, enable the built-in demo fallback. This allows the optimization
and dashboard to run, but be transparent that the scenario was loaded from the demo data
rather than interpreted by an LLM.

## Method in simple terms

### LLM layer
The LLM turns the user's free-text mission brief into structured JSON. It does not calculate
the optimal allocation or invent the reported optimization metrics.

### PSO layer
Each particle contains two sets of continuous keys:
- Assignment keys: decoded into a UAV index for every task.
- Ordering keys: sorted to determine the visit order of tasks assigned to each UAV.

The PSO updates particle velocity and position using inertia, personal-best and global-best
terms. The best candidate is selected by minimizing the weighted fitness.

### Path-planning layer
For each candidate allocation and visit order, A* finds a grid path from the UAV start to
its first task, then between subsequent tasks while avoiding obstacle cells.

### Fitness function
Lower is better:

`F = distance + 1.5*energy + 3*makespan + 2*workload_imbalance + 0.8*priority_delay + 120*collision_events + battery_penalty + infeasible_path_penalty`

Weights are illustrative demonstration parameters, not experimentally validated constants.

## Limitations to state clearly

- 2D grid world, not 3D flight dynamics.
- Static obstacles, not moving obstacles or changing weather.
- Simplified energy model: distance × UAV energy-per-cell.
- Collision risk is estimated by same-cell occupancy at discrete route steps; it is not a certified
  continuous-time separation guarantee.
- The PSO uses a random-key encoding and a weighted single fitness score, rather than a formal
  Pareto-front multi-objective PSO.
- Results depend on the random seed, particle count, iterations and chosen weights.
- No real UAV hardware, GPS, camera or flight controller is connected.

## Suggested research extension

Compare a simple baseline (nearest-task allocation) against PSO using the same scenario and
report distance, energy, makespan and fitness. This gives the demonstration an experimental
comparison rather than only a visual output.


## Cloud deployment (optional)

This project supports both local Ollama and hosted Groq inference:
- Without `GROQ_API_KEY`, the app uses local Ollama (`qwen2.5:3b`).
- With `GROQ_API_KEY` configured in Streamlit Community Cloud secrets, the app uses Groq.
- The PSO and A* algorithms always run in the Python application.

For Streamlit Community Cloud, add this in App settings → Secrets:

```toml
GROQ_API_KEY = "your-secret-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

Never commit the API key to GitHub. Deploy `app.py` from the repository root and ensure
`requirements.txt`, `optimizer.py`, and `llm_layer.py` are committed.

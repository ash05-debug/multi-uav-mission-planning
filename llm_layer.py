import json
import os
import requests

SYSTEM_PROMPT = """
You are a mission-brief parser for a multi-UAV disaster-response research demo.
Convert the user's natural-language scenario into valid JSON only, with this schema:
{
 "mission_name": string,
 "uavs": [{"name": string, "start": [integer,integer], "battery": number,
           "speed": number, "energy_per_cell": number}],
 "tasks": [{"name": string, "pos": [integer,integer], "priority": number}],
 "obstacles": [[integer,integer], ...]
}
Grid coordinates must be within x=0..19 and y=0..14. Use 2 to 4 UAVs and 4 to 10 tasks.
Priority is 1 (normal) to 5 (critical). Battery should be a positive energy budget.
Speed should be between 0.5 and 2.0 grid cells per time unit.
Use reasonable values if the brief omits exact values. Keep obstacles away from task
locations and UAV starting positions. Return JSON only, without markdown fences.
"""

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

def _groq_key():
    # Streamlit Community Cloud secrets, then environment variable for local testing.
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")

def _chat_groq(messages, model=None, json_mode=False):
    key = _groq_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to Streamlit secrets or your local environment.")
    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    payload = {"model": model, "messages": messages, "temperature": 0.1}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = requests.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90
    )
    if not response.ok:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:500]}")
    return response.json()["choices"][0]["message"]["content"]

def _chat_ollama(messages, model="qwen2.5:3b", base_url="http://localhost:11434", json_mode=False):
    payload = {"model": model, "stream": False, "messages": messages,
               "options": {"temperature": 0.1}}
    if json_mode:
        payload["format"] = "json"
    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]

def _chat(messages, model=None, json_mode=False, base_url="http://localhost:11434"):
    # If a Groq key is configured, use the hosted model; otherwise preserve local Ollama.
    if _groq_key():
        return _chat_groq(messages, model=model, json_mode=json_mode)
    return _chat_ollama(messages, model=model or "qwen2.5:3b",
                         base_url=base_url, json_mode=json_mode)

def parse_with_ollama(prompt, model="qwen2.5:3b", base_url="http://localhost:11434"):
    content = _chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        model=model,
        json_mode=True,
        base_url=base_url
    )
    data = json.loads(content)
    validate_scenario(data)
    return data

def validate_scenario(data):
    if not isinstance(data.get("uavs"), list) or not 2 <= len(data["uavs"]) <= 4:
        raise ValueError("LLM output must contain 2–4 UAVs.")
    if not isinstance(data.get("tasks"), list) or not 4 <= len(data["tasks"]) <= 10:
        raise ValueError("LLM output must contain 4–10 tasks.")
    for u in data["uavs"]:
        x, y = u["start"]
        if not (0 <= int(x) < 20 and 0 <= int(y) < 15):
            raise ValueError("A UAV start coordinate is outside the map.")
        if float(u["battery"]) <= 0 or float(u["speed"]) <= 0:
            raise ValueError("Battery and speed must be positive.")
    for t in data["tasks"]:
        x, y = t["pos"]
        if not (0 <= int(x) < 20 and 0 <= int(y) < 15):
            raise ValueError("A task coordinate is outside the map.")
        t["priority"] = max(1, min(5, int(t["priority"])))
    for o in data.get("obstacles", []):
        x, y = o
        if not (0 <= int(x) < 20 and 0 <= int(y) < 15):
            raise ValueError("An obstacle coordinate is outside the map.")
    occupied = {tuple(u["start"]) for u in data["uavs"]}
    occupied.update(tuple(t["pos"]) for t in data["tasks"])
    if any(tuple(o) in occupied for o in data.get("obstacles", [])):
        raise ValueError("An obstacle overlaps a UAV start or task location.")
    return data

def explain_result_with_ollama(scenario, result, model="qwen2.5:3b",
                               base_url="http://localhost:11434"):
    summary = {
        "mission": scenario.get("mission_name", "Disaster response"),
        "uavs": scenario["uavs"],
        "tasks": scenario["tasks"],
        "allocation": {
            scenario["uavs"][i]["name"]: [scenario["tasks"][t]["name"] for t in ids]
            for i, ids in enumerate(result["allocation"])
        },
        "metrics": {
            "distance": round(result["distance"], 2),
            "energy": round(result["energy"], 2),
            "completion_time": round(result["makespan"], 2),
            "workload_imbalance": round(result["imbalance"], 2),
            "collision_risk_events": result["collisions"],
            "fitness": round(result["fitness"], 2)
        }
    }
    prompt = (
        "Explain these actual optimization results for a university project demonstration. "
        "Do not invent numbers. Explain the allocation, trade-offs, and what the fitness "
        "score means in simple language. Mention that this is a 2D simulation, not a real flight.\n"
        + json.dumps(summary)
    )
    content = _chat(
        [{"role": "system", "content": "You explain technical simulation results accurately and clearly."},
         {"role": "user", "content": prompt}],
        model=model,
        json_mode=False,
        base_url=base_url
    )
    return content.strip()

# runner3.py
# DSPy Predict mit Tools aus OpenAPI
#
# Unterschied zu runner2.py:
# - Tools werden automatisch aus OpenAPI Spec erstellt
# - Tool-Beschreibungen kommen aus OpenAPI (nicht hardcoded)
#
# Gleich wie runner2.py:
# - dspy.Predict für einzelne Entscheidungen
# - Manueller Loop + Tool-Execution

import json
import uuid
from typing import List, Dict, Any

import dspy
import requests


# =============================================================================
# OpenAPI -> Tool-Definitionen (KEIN dspy.Tool, nur Daten)
# =============================================================================
def fetch_openapi_spec(base_url: str) -> dict:
    """Lade OpenAPI Spec vom Server."""
    r = requests.get(f"{base_url}/openapi.json", timeout=10)
    r.raise_for_status()
    return r.json()


def resolve_ref(spec: dict, ref: str) -> dict:
    """Löse $ref Referenz auf."""
    parts = ref.split("/")[1:]
    result = spec
    for part in parts:
        result = result[part]
    return result


def extract_tools_from_openapi(base_url: str, include_ops: List[str] = None) -> Dict[str, dict]:
    """
    Extrahiere Tool-Definitionen aus OpenAPI.
    Gibt ein Dict zurück: {operation_id: {path, description, args}}
    """
    spec = fetch_openapi_spec(base_url)
    tools = {}
    
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method.lower() != "post":
                continue
            
            op_id = operation.get("operationId")
            if not op_id or (include_ops and op_id not in include_ops):
                continue
            
            # Beschreibung
            summary = operation.get("summary", "")
            description = operation.get("description", "")
            
            # Request Schema
            request_body = operation.get("requestBody", {})
            content = request_body.get("content", {}).get("application/json", {})
            schema = content.get("schema", {})
            if "$ref" in schema:
                schema = resolve_ref(spec, schema["$ref"])
            
            # Args extrahieren
            properties = schema.get("properties", {})
            args_info = []
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "string")
                prop_desc = prop_schema.get("description", "")
                args_info.append(f"{prop_name} ({prop_type}): {prop_desc}")
            
            tools[op_id] = {
                "path": path,
                "summary": summary,
                "description": description,
                "args": "\n    ".join(args_info),
            }
            print(f"  Tool: {op_id}")
    
    return tools


def format_tools_for_prompt(tools: Dict[str, dict]) -> str:
    """Formatiere Tools für den Prompt."""
    lines = []
    for name, info in tools.items():
        lines.append(f"- {name}: {info['summary']}")
        lines.append(f"    {info['description']}")
        lines.append(f"    Args: {info['args']}")
    return "\n".join(lines)


def call_tool(base_url: str, tools: Dict[str, dict], tool_name: str, args: dict) -> dict:
    """Führe Tool aus via HTTP."""
    if tool_name not in tools:
        return {"error": f"Unknown tool: {tool_name}"}
    
    path = tools[tool_name]["path"]
    payload = {k: v for k, v in args.items() if v is not None}
    
    try:
        r = requests.post(f"{base_url}{path}", json=payload, timeout=10)
        return {"http": r.status_code, "json": r.json()}
    except Exception as e:
        return {"http": 500, "json": {"error": str(e)}}


# =============================================================================
# DSPy Signature (wie runner2.py)
# =============================================================================
class ToolCallSignature(dspy.Signature):
    """Decide the NEXT SINGLE tool call based on what you already know.
    Output exactly ONE JSON object per turn.

    WORKFLOW:
    1. Need ISO2 country code? -> resolve_country(name="Deutschland") -> returns "DE"
    2. Have ISO2, need postal code? -> resolve_postal_code(country="DE", city="Berlin") -> returns "10115"  
    3. Have ISO2 AND postal code? -> get_shipping_quote(country="DE", postal_code="10115", ...)

    CRITICAL: Check tool_results! Use the VALUES you received, e.g.:
    - If tool_results shows {"iso2":"DE"} -> use country="DE" (not "resolve_country")
    - If tool_results shows {"postal_code":"10115"} -> use postal_code="10115"
    """
    user_request: str = dspy.InputField(desc="The user's shipping request")
    available_tools: str = dspy.InputField(desc="Available tools")
    tool_results: str = dspy.InputField(desc="Previous results - extract values from here!")
    
    json_output: str = dspy.OutputField(desc="Exactly ONE JSON object: {tool_name, args}")


# =============================================================================
# JSON Extraction (wie runner.py/runner2.py)
# =============================================================================
def extract_json_objects(text: str) -> List[dict]:
    """Extrahiere JSON-Objekte aus Text."""
    objs = []
    start = None
    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
        elif in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    objs.append(json.loads(text[start:i+1]))
                except:
                    pass
                start = None
    return objs


# =============================================================================
# Hauptlogik (wie runner2.py)
# =============================================================================
def run_agentic(user_text: str, base_url: str, tools: Dict[str, dict], max_rounds: int = 3):
    predictor = dspy.Predict(ToolCallSignature)
    tools_desc = format_tools_for_prompt(tools)
    
    tool_results_so_far = ""
    quotes = []
    tool_history = []

    for round_idx in range(1, max_rounds + 1):
        result = predictor(
            user_request=user_text,
            available_tools=tools_desc,
            tool_results=tool_results_so_far if tool_results_so_far else "(keine bisherigen Aufrufe)"
        )
        raw = result.json_output
        objs = extract_json_objects(raw)
        
        tool_results_this_round = []

        for obj in objs:
            tool_name = obj.get("tool_name")
            args = obj.get("args", {})
            
            if not tool_name:
                continue
            
            res = call_tool(base_url, tools, tool_name, args)
            
            event = {"tool_name": tool_name, "args": args, "result": res}
            tool_history.append(event)
            tool_results_this_round.append(event)

            if tool_name == "get_shipping_quote" and res.get("http") == 200:
                quotes.append({"args": args, "response": res["json"]})

        # Prüfe ob ALLE Quotes erfolgreich
        quote_calls = [e for e in tool_results_this_round if e.get("tool_name") == "get_shipping_quote"]
        quote_ok = [q for q in quote_calls if q.get("result", {}).get("http") == 200]
        
        if quote_calls and len(quote_ok) == len(quote_calls):
            return {"ok": True, "quotes": quotes, "rounds_used": round_idx, "tool_history": tool_history}

        tool_results_so_far += f"\nRound {round_idx}:\n{json.dumps(tool_results_this_round, ensure_ascii=False)}"

    return {"ok": False, "error": "max_rounds", "rounds_used": max_rounds, "tool_history": tool_history}


# =============================================================================
# Ausgabe
# =============================================================================
def print_compact(prompt: str, result: dict) -> None:
    print(f"TestCase: {prompt}")
    
    for call in result.get("tool_history", []):
        tool = call["tool_name"]
        args = call["args"]
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"  LLM calls {tool}({args_str})")
        
        res = call.get("result", {})
        http = res.get("http", "?")
        json_res = res.get("json", {})
        if "error" in json_res and json_res["error"]:
            print(f"  API Result: HTTP {http} -> error={json_res['error']}")
        elif "price" in json_res:
            print(f"  API Result: HTTP {http} -> {json_res['price']} {json_res.get('currency', 'EUR')}")
        elif "iso2" in json_res:
            print(f"  API Result: HTTP {http} -> {json_res['iso2']}")
        elif "postal_code" in json_res:
            print(f"  API Result: HTTP {http} -> {json_res['postal_code']} ({json_res.get('city', '')})")
        else:
            print(f"  API Result: HTTP {http} -> {json_res}")
    
    status = "OK" if result.get("ok") else f"FAILED: {result.get('error', 'unknown')}"
    rounds = result.get("rounds_used", "?")
    quotes_count = len(result.get("quotes", []))
    print(f"  => {status}, {quotes_count} quote(s), {rounds} round(s)")
    print("-" * 60)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    OPENAPI_BASE = "http://localhost:9000"
    
    # LM konfigurieren
    lm = dspy.LM(
        model="openai/meta-llama/Llama-3.1-8B-Instruct",
        api_base="http://localhost:8000/v1",
        api_key="local",
    )
    dspy.configure(lm=lm)
    
    # Tools aus OpenAPI laden
    print("Loading tools from OpenAPI...")
    tools = extract_tools_from_openapi(
        OPENAPI_BASE,
        include_ops=["resolve_country", "resolve_postal_code", "get_shipping_quote"]
    )
    print(f"Loaded {len(tools)} tools\n")
    
    tests = [
        "Schick das nach Deutschland, Berlin, 1 kg, express.",
        "Einmal nach Deutschland Berlin 1 kg express und einmal nach Österreich Wien 1 kg standard.",
        "Versand nach FR Paris 2 kg standard.",
    ]

    for t in tests:
        out = run_agentic(t, OPENAPI_BASE, tools)
        print_compact(t, out)

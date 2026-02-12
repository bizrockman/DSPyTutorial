import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Literal

import dspy
import requests
from pydantic import BaseModel, ValidationError, Field


def log_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append a JSON object as a single line to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------- JSON object stream extraction -----------------
def extract_json_objects(text: str) -> List[dict]:
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
                in_str = False
                esc = False
            continue

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start:i+1]
                try:
                    objs.append(json.loads(chunk))
                except json.JSONDecodeError:
                    pass
                start = None

    return objs


# ----------------- OpenAI-compatible chat/completions -----------------
def chat(api_base: str, api_key: str, model: str, messages: list, temperature=0.0, max_tokens=500) -> str:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    return msg.get("content") or ""


# ----------------- ToolCall schema -----------------
class ResolverCountryArgs(BaseModel):
    name: str = Field(min_length=2)

class ResolverPostalArgs(BaseModel):
    country: Optional[str] = None
    city: Optional[str] = None
    value: Optional[str] = None
    mode: Literal["lookup_city", "validate_postal"] = "lookup_city"

class QuoteArgs(BaseModel):
    country: str = Field(min_length=2, max_length=2)
    postal_code: str = Field(min_length=3, max_length=12)
    weight_kg: float = Field(gt=0.0, lt=50.0)
    service: Literal["standard", "express"] = "standard"

class ToolCall(BaseModel):
    tool_name: Literal["resolve_country", "resolve_postal_code", "get_shipping_quote"]
    args: Dict[str, Any]


def validate_toolcall(obj: dict) -> Tuple[Optional[ToolCall], str]:
    try:
        tc = ToolCall.model_validate(obj)
        return tc, ""
    except ValidationError as e:
        return None, f"toolcall_schema_error: {e.errors()[:2]}"


# ----------------- Tool execution -----------------
def call_tool(openapi_base: str, trace_id: str, tool_name: str, args: dict) -> dict:
    if tool_name == "resolve_country":
        path = "/v1/resolve/country"
    elif tool_name == "resolve_postal_code":
        path = "/v1/resolve/postal"
    elif tool_name == "get_shipping_quote":
        path = "/v1/shipping/quote"
    else:
        return {"http": 400, "json": {"error": "unknown_tool"}}

    url = f"{openapi_base.rstrip('/')}{path}"
    r = requests.post(url, json=args, headers={"x-trace-id": trace_id}, timeout=10)
    try:
        return {"http": r.status_code, "json": r.json()}
    except Exception:
        return {"http": r.status_code, "json": {"error": "non-json-response", "text": r.text[:500]}}


class ShippingQuoteSignature(dspy.Signature):
    """You are a shipping quote agent. Your goal: provide accurate shipping quotes for the user.

    WORKFLOW:
    1. If you see a country NAME (like "Germany", "Deutschland") -> call resolve_country to get ISO2 code
    2. If you have ISO2 but no postal code -> call resolve_postal_code(mode="lookup_city", country="DE", city="Berlin")
    3. Once you have ISO2 country AND postal code -> call get_shipping_quote

    OUTPUT FORMAT:
    Output ONLY JSON objects (only multiple get_shipping_quote objects are allowed otherwise one object ONLY. Call Resolver 
    only if needed, if you know all arguments for get_shipping_quote, call it directly.). No text outside JSON.

    AVAILABLE TOOLS:
    - resolve_country: {"tool_name":"resolve_country","args":{"name":"Deutschland"}}
    - resolve_postal_code: {"tool_name":"resolve_postal_code","args":{"mode":"lookup_city","country":"DE","city":"Berlin"}}
    - get_shipping_quote: {"tool_name":"get_shipping_quote","args":{"country":"DE","postal_code":"10115","weight_kg":1.0,"service":"express"}}

    RULES:
    - Do NOT guess postal codes or country codes. Use resolver tools first.
    - For multiple shipments, process them step by step.
    - Always use the VALUES from previous tool results (e.g., if resolve_country returned "DE", use country="DE").
    - You have at most 3 rounds to produce at least one successful get_shipping_quote call.
    """
    user_request: str = dspy.InputField(desc="The user's shipping request")
    tool_results: str = dspy.InputField(desc="Results from previous tool calls (empty if first round)")
    
    json_output: str = dspy.OutputField(desc="One or more JSON tool call objects")


# Set the system prompt here
def run_agentic(user_text: str, openapi_base: str, max_rounds: int = 3):    

    trace_id = str(uuid.uuid4())
    predictor = dspy.Predict(ShippingQuoteSignature)
    
    tool_results_so_far = ""
    quotes: List[dict] = []
    tool_history: List[dict] = []

    for round_idx in range(1, max_rounds + 1):
        result = predictor(
            user_request=user_text,
            tool_results=tool_results_so_far if tool_results_so_far else "(keine bisherigen Aufrufe)"
        )
        raw = result.json_output
        
        objs = extract_json_objects(raw) 
        tool_results_this_round = []        
        bad_objects = 0

        # Execute all toolcalls in the order they appear
        for obj in objs:
            tc, err = validate_toolcall(obj)
            if tc is None:
                bad_objects += 1
                tool_results_this_round.append({"error": err, "raw_obj": obj})
                continue

            # Validate args per tool (optional but recommended)
            if tc.tool_name == "resolve_country":
                try:
                    ResolverCountryArgs.model_validate(tc.args)
                except ValidationError as e:
                    tool_results_this_round.append({"tool": "resolve_country", "error": e.errors()[:2], "args": tc.args})
                    continue

            elif tc.tool_name == "resolve_postal_code":
                try:
                    ResolverPostalArgs.model_validate(tc.args)
                except ValidationError as e:
                    tool_results_this_round.append({"tool": "resolve_postal_code", "error": e.errors()[:2], "args": tc.args})
                    continue

            elif tc.tool_name == "get_shipping_quote":
                try:
                    QuoteArgs.model_validate(tc.args)
                except ValidationError as e:
                    tool_results_this_round.append({"tool": "get_shipping_quote", "error": e.errors()[:2], "args": tc.args})
                    continue
            else:
                tool_results_this_round.append({"error": "unknown_tool", "raw_obj": obj})
                continue

            res = call_tool(openapi_base, trace_id, tc.tool_name, tc.args)

            event = {"tool_name": tc.tool_name, "args": tc.args, "result": res, "round": round_idx}
            tool_history.append(event)
            tool_results_this_round.append(event)

            if tc.tool_name == "get_shipping_quote" and res.get("http") == 200:
                quotes.append({"args": tc.args, "response": res["json"]})

        # If we got at least one successful quote, stop: goal reached
        if quotes:
            return {
                "trace_id": trace_id,
                "ok": True,
                "quotes": quotes,
                "rounds_used": round_idx,
                "tool_history": tool_history,
                "raw_last": raw,
            }

        tool_results_so_far += f"\nRound {round_idx}:\n{json.dumps(tool_results_this_round, ensure_ascii=False)}"

    return {
        "trace_id": trace_id,
        "ok": False,
        "error": "no_quote_within_max_rounds",
        "rounds_used": max_rounds,
        "tool_history": tool_history,
    }


def print_compact(prompt: str, result: dict) -> None:
    """Compact output for tests with grouping by rounds."""
    print(f"TestCase: {prompt}")
    
    current_round = None
    for call in result.get("tool_history", []):
        # Show round header if new round starts
        r = call.get("round")
        if r is not None and r != current_round:
            current_round = r
            print(f"  Round {r}:")
        
        tool = call["tool_name"]
        args = call["args"]
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"    LLM calls {tool}({args_str})")
        
        res = call.get("result", {})
        http = res.get("http", "?")
        json_res = res.get("json", {})
        # Compact representation of the result
        if "error" in json_res and json_res["error"]:
            print(f"    API Result: HTTP {http} -> error={json_res['error']}")
        elif "price" in json_res:
            print(f"    API Result: HTTP {http} -> {json_res['price']} {json_res.get('currency', 'EUR')}")
        elif "iso2" in json_res:
            print(f"    API Result: HTTP {http} -> {json_res['iso2']}")
        elif "postal_code" in json_res:
            print(f"    API Result: HTTP {http} -> {json_res['postal_code']} ({json_res.get('city', '')})")
        else:
            print(f"    API Result: HTTP {http} -> {json_res}")
    
    status = "OK" if result.get("ok") else f"FAILED: {result.get('error', 'unknown')}"
    rounds = result.get("rounds_used", "?")
    quotes = len(result.get("quotes", []))
    print(f"  => {status}, {quotes} quote(s), {rounds} round(s)")
    print("-" * 60)


if __name__ == "__main__":
    lm = dspy.LM(
        model="openai/meta-llama/Llama-3.1-8B-Instruct",
        api_base="http://localhost:8000/v1",
        api_key="local",
    )
    dspy.configure(lm=lm)

    OPENAPI_BASE = "http://localhost:9000"

    tests = [
        "Schick das nach Deutschland, Berlin, 1 kg, express.",
        "Einmal nach Deutschland Borken 1 kg express und einmal nach Österreich Wien 1 kg standard.",
        "Versand nach FR Paris 2 kg standard.",
    ]

    for t in tests:
        out = run_agentic(t, OPENAPI_BASE)
        print_compact(t, out)
        # For debugging: log the complete JSON output to a file
        log_jsonl("runner2_runs.jsonl", out)
        # print("\n[DEBUG] Last LLM-Call:")
        # lm.inspect_history(n=1)

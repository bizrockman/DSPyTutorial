# dspy_agent2.py
# DSPy Agent mit automatischer Tool-Generierung aus OpenAPI Spec
import dspy
import requests
from typing import Any, Dict, List, Callable


# =============================================================================
# OpenAPI -> DSPy Tools Konvertierung
# =============================================================================
def fetch_openapi_spec(base_url: str) -> dict:
    """Lade OpenAPI Spec vom Server."""
    r = requests.get(f"{base_url}/openapi.json", timeout=10)
    r.raise_for_status()
    return r.json()


def resolve_schema_ref(spec: dict, schema: dict) -> dict:
    """Löse $ref Referenzen in OpenAPI Schema auf."""
    if "$ref" in schema:
        ref_path = schema["$ref"]  # z.B. "#/components/schemas/ShippingQuoteRequest"
        parts = ref_path.split("/")
        resolved = spec
        for part in parts[1:]:  # Skip '#'
            resolved = resolved[part]
        return resolved
    return schema


def extract_args_from_schema(spec: dict, schema: dict) -> tuple[dict, dict]:
    """
    Extrahiere args und arg_desc aus OpenAPI Schema.
    Returns: (args_dict, arg_desc_dict)
    """
    resolved = resolve_schema_ref(spec, schema)
    properties = resolved.get("properties", {})
    required = resolved.get("required", [])
    
    args = {}
    arg_desc = {}
    
    for prop_name, prop_schema in properties.items():
        # Typ mapping OpenAPI -> JSON Schema
        prop_type = prop_schema.get("type", "string")
        
        arg_schema = {"type": prop_type}
        
        # Enum handling
        if "enum" in prop_schema:
            arg_schema["enum"] = prop_schema["enum"]
        
        # Default handling
        if "default" in prop_schema:
            arg_schema["default"] = prop_schema["default"]
        
        args[prop_name] = arg_schema
        
        # Beschreibung
        desc = prop_schema.get("description", "")
        if prop_name in required:
            desc = f"(required) {desc}" if desc else "(required)"
        arg_desc[prop_name] = desc
    
    return args, arg_desc


def create_tool_function(base_url: str, path: str, operation_id: str) -> Callable:
    """Erstelle eine Tool-Funktion für einen API-Endpunkt."""
    
    def tool_func(**kwargs) -> dict:
        # Entferne None-Werte
        payload = {k: v for k, v in kwargs.items() if v is not None}
        r = requests.post(f"{base_url}{path}", json=payload, timeout=10)
        return r.json()
    
    tool_func.__name__ = operation_id
    return tool_func


def create_tools_from_openapi(base_url: str, include_operations: List[str] = None) -> List[dspy.Tool]:
    """
    Erstelle DSPy Tools aus OpenAPI Spec.
    
    Args:
        base_url: Basis-URL des API-Servers
        include_operations: Liste von operationIds die inkludiert werden sollen (None = alle)
    
    Returns:
        Liste von dspy.Tool Objekten
    """
    spec = fetch_openapi_spec(base_url)
    tools = []
    
    for path, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method.lower() != "post":
                continue  # Nur POST-Endpunkte als Tools
            
            operation_id = operation.get("operationId")
            if not operation_id:
                continue
            
            # Filter nach include_operations
            if include_operations and operation_id not in include_operations:
                continue
            
            # Beschreibung
            summary = operation.get("summary", "")
            description = operation.get("description", "")
            full_desc = f"{summary}. {description}".strip(". ")
            
            # Request Schema
            request_body = operation.get("requestBody", {})
            content = request_body.get("content", {}).get("application/json", {})
            schema = content.get("schema", {})
            
            args, arg_desc = extract_args_from_schema(spec, schema)
            
            # Tool-Funktion erstellen
            func = create_tool_function(base_url, path, operation_id)
            
            # Docstring für bessere Tool-Beschreibung
            func.__doc__ = full_desc
            
            # DSPy Tool erstellen
            tool = dspy.Tool(
                func=func,
                name=operation_id,
                desc=full_desc,
                args=args,
                arg_desc=arg_desc,
            )
            
            tools.append(tool)
            print(f"  Created tool: {operation_id} ({path})")
    
    return tools


# =============================================================================
# DSPy Agent
# =============================================================================
class ShippingQuoteSignature(dspy.Signature):
    """
    You are a shipping quote agent. Your goal: provide accurate shipping quotes for the user.

    WORKFLOW:
    1. If you see a country NAME (like "Germany", "Deutschland") -> call resolve_country to get ISO2 code
    2. If you have ISO2 but no postal code -> call resolve_postal_code(mode="lookup_city", country="DE", city="Berlin")
    3. Once you have ISO2 country AND postal code -> call get_shipping_quote

    RULES:
    - Do NOT guess postal codes or country codes. Use resolver tools first.
    - For multiple shipments, process them step by step.
    - Always use the VALUES from previous tool results (e.g., if resolve_country returned "DE", use country="DE").
    - You have at most 3 rounds to produce at least one successful get_shipping_quote call.
    """
    user_request: str = dspy.InputField(desc="The user's shipping request")
    final_answer: str = dspy.OutputField(desc="Summary of all shipping quotes, or explanation if failed")


class OpenAPIAgent(dspy.Module):
    def __init__(self, tools: List[dspy.Tool], max_iters: int = 6):
        super().__init__()
        self.agent = dspy.ReAct(
            ShippingQuoteSignature,
            tools=tools,
            max_iters=max_iters,
        )

    def forward(self, user_request: str):
        return self.agent(user_request=user_request)


# =============================================================================
# Kompakte Ausgabe
# =============================================================================
def print_trajectory(result) -> None:
    """Zeige die Tool-Aufrufe im kompakten Format."""
    trajectory = getattr(result, 'trajectory', None)
    
    if trajectory and isinstance(trajectory, dict):
        i = 0
        while True:
            thought_key = f"thought_{i}"
            tool_name_key = f"tool_name_{i}"
            tool_args_key = f"tool_args_{i}"
            obs_key = f"observation_{i}"
            
            if tool_name_key not in trajectory:
                break
            
            thought = trajectory.get(thought_key, "")
            tool_name = trajectory.get(tool_name_key, "")
            tool_args = trajectory.get(tool_args_key, {})
            observation = trajectory.get(obs_key, "")
            
            if thought:
                t_str = str(thought)[:100] + "..." if len(str(thought)) > 100 else str(thought)
                print(f"  Thought: {t_str}")
            
            if tool_name:
                if isinstance(tool_args, dict):
                    args_str = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
                else:
                    args_str = str(tool_args)
                print(f"  -> {tool_name}({args_str})")
            
            if observation:
                obs_str = str(observation)[:100] + "..." if len(str(observation)) > 100 else str(observation)
                print(f"  <- {obs_str}")
            
            i += 1


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
    
    # Tools aus OpenAPI generieren
    print("Loading tools from OpenAPI spec...")
    tools = create_tools_from_openapi(
        OPENAPI_BASE,
        include_operations=["resolve_country", "resolve_postal_code", "get_shipping_quote"]
    )
    print(f"Loaded {len(tools)} tools\n")
    
    # Agent erstellen
    agent = OpenAPIAgent(tools=tools, max_iters=6)
    
    # Testfälle
    tests = [
        "Schick das nach Deutschland, Berlin, 1 kg, express.",
        "Einmal nach Deutschland Berlin 1 kg express und einmal nach Österreich Wien 1 kg standard.",
        "Versand nach FR Paris 2 kg standard.",
    ]
    
    for t in tests:
        print("=" * 60)
        print(f"TestCase: {t}")
        print("=" * 60)
        
        try:
            result = agent(user_request=t)
            print_trajectory(result)
            print(f"\n=> Final Answer: {result.final_answer}")
        except Exception as e:
            import traceback
            print(f"\nError: {e}")
            traceback.print_exc()
        
        print("-" * 60 + "\n")

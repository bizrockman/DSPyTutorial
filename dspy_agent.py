# dspy_agent.py
# Minimaler, sauberer DSPy ReAct Agent
import dspy
import requests
from pydantic import BaseModel, Field
from typing import Literal, Optional

# =============================================================================
# Konfiguration
# =============================================================================
OPENAPI_BASE = "http://localhost:9000"


# =============================================================================
# Tool-Funktionen (echte Python-Funktionen, die DSPy versteht)
# =============================================================================
def resolve_country(name: str) -> dict:
    """
    Convert a country NAME to ISO2 code.
    Use this when you have a country name like 'Germany', 'Deutschland', 'France'.
    Do NOT use if you already have a 2-letter code like DE, FR, AT.
    
    Args:
        name: The country name to resolve (e.g., 'Germany', 'Deutschland')
    
    Returns:
        dict with 'iso2' (the 2-letter code) or 'error' if not found
    """
    r = requests.post(f"{OPENAPI_BASE}/v1/resolve/country", json={"name": name}, timeout=10)
    return r.json()


def resolve_postal_code(mode: str, country: str, city: str = None, value: str = None) -> dict:
    """
    Look up a postal code for a city, or validate an existing postal code.
    
    Args:
        mode: Either 'lookup_city' (to find postal code for a city) or 'validate_postal'
        country: ISO2 country code (e.g., 'DE', 'FR', 'AT')
        city: City name (required when mode='lookup_city')
        value: Postal code to validate (required when mode='validate_postal')
    
    Returns:
        dict with 'postal_code' and 'city', or 'error' if not found
    """
    payload = {"mode": mode, "country": country}
    if city:
        payload["city"] = city
    if value:
        payload["value"] = value
    r = requests.post(f"{OPENAPI_BASE}/v1/resolve/postal", json=payload, timeout=10)
    return r.json()


def get_shipping_quote(country: str, postal_code: str, weight_kg: float, service: str = "standard") -> dict:
    """
    Calculate shipping quote. This is the FINAL step after you have ISO2 country and postal code.
    
    Args:
        country: ISO2 country code (e.g., 'DE')
        postal_code: Valid postal code for the destination
        weight_kg: Package weight in kilograms (must be > 0 and < 50)
        service: Either 'standard' or 'express'
    
    Returns:
        dict with 'price', 'currency', and 'service'
    """
    r = requests.post(
        f"{OPENAPI_BASE}/v1/shipping/quote",
        json={"country": country, "postal_code": postal_code, "weight_kg": weight_kg, "service": service},
        timeout=10,
    )
    return r.json()


# =============================================================================
# DSPy Signature (beschreibt Ziel und Workflow)
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


# =============================================================================
# Runner
# =============================================================================
if __name__ == "__main__":
    # LM konfigurieren
    lm = dspy.LM(
        model="openai/meta-llama/Llama-3.1-8B-Instruct",
        api_base="http://localhost:8000/v1",
        api_key="local",
    )
    dspy.configure(lm=lm)
    
    agent = dspy.ReAct(
        ShippingQuoteSignature,
        tools=[resolve_country, resolve_postal_code, get_shipping_quote],
        max_iters=6,
    )

    # Testfälle
    tests = [
        "Schick das nach Deutschland, Berlin, 1 kg, express.",
        "Einmal nach Deutschland Borken 1 kg express und einmal nach Österreich Wien 1 kg standard.",
        "Versand nach FR Paris 2 kg standard.",
    ]

    for t in tests:
        print("\n" + "=" * 60)
        print(f"TestCase: {t}")
        print("=" * 60)
        
        try:
            result = agent(user_request=t)
            
            # Zeige die Trajectory
            trajectory = getattr(result, 'trajectory', None)
            
            if trajectory and isinstance(trajectory, dict):
                # Trajectory ist ein Dict mit Keys wie thought_0, tool_name_0, etc.
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
                    
                    # Kompakte Ausgabe
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
                    
            elif trajectory:
                print(f"  [Debug] Trajectory: {trajectory}")
            else:
                print("  [Debug] No trajectory found")
            
            print(f"\n=> Final Answer: {result.final_answer}")
            
        except Exception as e:
            import traceback
            print(f"\nError: {e}")
            traceback.print_exc()
        
        print("-" * 60)

# DSPy Tutorial: Building Reliable Agents

Dieses Repository begleitet eine Artikelserie über den Übergang von manuellem Prompt Engineering zu DSPy. Es demonstriert schrittweise, wie man einen robusten Agenten für Versandkostenanfragen baut – von einem einfachen Skript bis hin zu einem optimierten DSPy-Modul.

## Setup

1. **Python Umgebung erstellen:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

2. **Abhängigkeiten installieren:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lokalen LLM Server starten:**
   Die Skripte sind für einen lokalen LLM-Server konfiguriert, der eine OpenAI-kompatible API auf `http://localhost:8000` bereitstellt (z.B. [vLLM](https://docs.vllm.ai/), [Llama.cpp](https://github.com/ggml-org/llama.cpp) oder [Ollama](https://ollama.com/)).
   
   *Warum lokal?* Große Frontier-Modelle (GPT-4o, Claude 3.5) lösen einfache Tool-Calling-Aufgaben oft "zu gut" – sie rüberspringen Resolver-Schritte und verdecken damit die Herausforderungen, die bei der Agenten-Entwicklung in der Praxis auftreten. Kleinere Modelle (wie Llama 3.1 8B) machen diese Probleme sichtbar und zeigen, wie DSPy gezielt dabei hilft, auch mit begrenzten Modellen zuverlässige Agenten zu bauen.

   *Andere Modelle nutzen:* DSPy verwendet intern [LiteLLM](https://docs.litellm.ai/docs/providers) und unterstützt damit über 100 Provider. Die Konfiguration folgt dem Schema `"{provider}/{model}"`:
   ```python
   # Lokal (OpenAI-kompatibel, z.B. vLLM)
   lm = dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct", api_base="http://localhost:8000/v1", api_key="local")

   # OpenAI
   lm = dspy.LM("openai/gpt-4o", api_key="sk-...")

   # Anthropic
   lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022", api_key="sk-ant-...")

   dspy.configure(lm=lm)
   ```
   Siehe auch: [DSPy Language Models Dokumentation](https://dspy.ai/learn/programming/language_models/)

4. **Mock-API starten:**
   Die Skripte greifen auf eine lokale FastAPI-Anwendung zu (`main.py`), die Versand-APIs simuliert.
   ```bash
   uvicorn main:app --port 9000
   ```

## Die Evolution des Agenten

Das Tutorial folgt einer klaren Evolution:

### 1. Der manuelle Ansatz (`runner.py`)
- **Konzept:** Klassisches Prompt Engineering.
- **Technik:** Ein großer `SYSTEM`-Prompt beschreibt Tools und Regeln. Manuelle `requests`-Aufrufe an das LLM.
- **Problem:** Fragil, schwer zu warten, Prompt-Änderungen haben unvorhersehbare Effekte.

### 2. Erste Schritte mit DSPy (`runner2.py`)
- **Konzept:** Deklarative Signaturen statt Prompt-Strings.
- **Technik:** `dspy.Signature` definiert Input/Output. `dspy.Predict` ersetzt den manuellen LLM-Aufruf.
- **Unterschied:** Die Logik (Loop, JSON-Parsing) bleibt gleich, aber der Prompt ist nun strukturiert und typisiert.

### 3. Dynamische Tools (`runner3.py`)
- **Konzept:** Tools werden nicht mehr hardcodiert, sondern dynamisch aus einer OpenAPI-Spezifikation geladen.
- **Technik:** Automatische Generierung von Tool-Beschreibungen für den Prompt.
- **Lektion:** "Mehr Prompt ≠ Bessere Ergebnisse". Das Modell wird durch die ausführlichen OpenAPI-Beschreibungen verwirrt und die Performance sinkt.

### 4. ReAct Agent (`dspy_agent.py`)
- **Konzept:** Nutzung von DSPy's eingebautem `ReAct` Modul.
- **Technik:** Statt einer manuellen Schleife nutzen wir `dspy.ReAct`. Das Modell entscheidet selbstständig über Thought-Action-Observation-Schritte.
- **Vorteil:** Robusteres Reasoning, aber höhere Latenz/Kosten durch viele Roundtrips.

### 5. ReAct mit OpenAPI (`dspy_agent2.py`)
- **Konzept:** Kombination aus ReAct und dynamischen OpenAPI-Tools.
- **Technik:** Tools werden aus OpenAPI geladen und direkt als ausführbare Funktionen an `dspy.ReAct` übergeben.
- **Ergebnis:** Ein voll dynamischer Agent, der sich an API-Änderungen anpasst.

## Ausblick

Im nächsten Teil der Serie werden wir zeigen, wie man die Performance von komplexen Agenten durch automatische Optimierung (`BootstrapFewShot`, `MIPROv2`) signifikant verbessern kann.

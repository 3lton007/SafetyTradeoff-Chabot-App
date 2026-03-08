from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
# Get your endpoint URL from: HuggingFace → Inference Endpoints → your endpoint → Overview
# Get your token from: HuggingFace → Settings → Access Tokens
HF_ENDPOINT   = os.getenv("HF_ENDPOINT")   # e.g. https://xyz123.us-east-1.aws.endpoints.huggingface.cloud
HF_TOKEN      = os.getenv("HF_TOKEN")       # Your HuggingFace token (read token is enough)
MODEL_DISPLAY_NAME = "Safety AI"
MAX_NEW_TOKENS     = 300
# ──────────────────────────────────────────────────────────────────────────────

if not HF_ENDPOINT:
    print("[WARN] HF_ENDPOINT is not set — requests will fail!")
else:
    print(f"[INFO] Using HF endpoint: {HF_ENDPOINT}")
if not HF_TOKEN:
    print("[WARN] HF_TOKEN is not set — requests will fail!")
else:
    print("[INFO] HF token loaded.")

# In-memory conversation history (single user)
conversation_history = []


def build_prompt(history: list[dict]) -> str:
    """
    Formats conversation history into the prompt format the model was trained with.
    Uses simple User/Assistant format matching the safety fine-tuning data.
    Note: no <|begin_of_text|> prefix — that is LLaMA-only, not Gemma.
    """
    prompt = ""
    for msg in history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        else:
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant:"
    return prompt


@app.route("/")
def index():
    return render_template("chat.html", model_name=MODEL_DISPLAY_NAME)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "Empty message"}), 400

    user_message = data["message"].strip()
    conversation_history.append({"role": "user", "content": user_message})

    prompt = build_prompt(conversation_history)

    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.7,
            },
        }
        response = requests.post(HF_ENDPOINT, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return jsonify({"error": f"Endpoint error: {response.status_code} {response.text}"}), 500

        result = response.json()

        # handler.py returns: [{"generated_text": "..."}]
        if isinstance(result, list) and result:
            assistant_message = result[0].get("generated_text", "").strip()
        else:
            assistant_message = str(result)

    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out — please try again."}), 503
    except Exception as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500

    conversation_history.append({"role": "assistant", "content": assistant_message})
    return jsonify({"response": assistant_message})


@app.route("/clear", methods=["POST"])
def clear_chat():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "ok"})


@app.route("/load_session", methods=["POST"])
def load_session():
    global conversation_history
    data = request.get_json(silent=True)
    if data and isinstance(data.get("messages"), list):
        conversation_history = data["messages"]
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

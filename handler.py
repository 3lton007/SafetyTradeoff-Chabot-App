from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
import json
import os

# Patch JSON encoder to handle torch.dtype objects.
# The server's transformers version converts "float16" in config.json to
# torch.float16 (a dtype object), which json.dumps() cannot serialize.
_original_default = json.JSONEncoder.default
def _patched_default(self, obj):
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    return _original_default(self, obj)
json.JSONEncoder.default = _patched_default


class StopOnTokens(StoppingCriteria):
    """
    Stops generation when:
      - An EOS or <end_of_turn> token is generated (single-token stop)
      - The last N tokens match the "\nUser" sequence (multi-token stop)
        This prevents the model from starting a new fake user turn and
        ensures "User" never leaks into the decoded output.
    """
    def __init__(self, stop_ids: set, newline_user_ids: list):
        self.stop_ids = stop_ids
        self.newline_user_ids = newline_user_ids

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        last = input_ids[0][-1].item()
        if last in self.stop_ids:
            return True
        seq = input_ids[0][-len(self.newline_user_ids):].tolist()
        if seq == self.newline_user_ids:
            return True
        return False


class EndpointHandler:
    """
    Custom HuggingFace Inference Endpoint handler for Gemma 2 9B safety model.
    Loaded once at startup — all requests go through __call__.

    Expected request format:
        {"inputs": "your message here"}

    Optional parameters (passed inside request JSON):
        {"inputs": "...", "parameters": {"max_new_tokens": 300, "temperature": 0.7}}

    Response format:
        [{"generated_text": "assistant response"}]
    """

    def __init__(self, path: str = ""):
        # Fix tokenizer_config.json if extra_special_tokens is a list instead of dict.
        # This is a known Gemma tokenizer bug with certain transformers versions.
        tokenizer_config_path = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r") as f:
                config = json.load(f)
            if isinstance(config.get("extra_special_tokens"), list):
                config["extra_special_tokens"] = {}
                with open(tokenizer_config_path, "w") as f:
                    json.dump(config, f)
                print("[INFO] Patched extra_special_tokens in tokenizer_config.json")

        print("[INFO] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        print("[INFO] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,  # server uses older transformers — torch_dtype is correct here
            device_map="auto",
        )
        self.model.eval()
        print("[INFO] Model ready.")

        # ── Build stop token IDs once at startup ──────────────────────────────
        stop_ids = {self.tokenizer.eos_token_id}

        end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if end_of_turn_id != self.tokenizer.unk_token_id:
            stop_ids.add(end_of_turn_id)

        self.stop_ids = stop_ids
        self.newline_user_ids = self.tokenizer.encode("\nUser", add_special_tokens=False)
        print(f"[INFO] Stop IDs: {self.stop_ids}")
        print(f"[INFO] \\nUser token IDs: {self.newline_user_ids}")

    def __call__(self, data: dict) -> list:
        # ── Parse request ──────────────────────────────────────────────────────
        message = data.get("inputs", "")
        if not message or not message.strip():
            return [{"generated_text": ""}]

        params = data.get("parameters", {})
        max_new_tokens = params.get("max_new_tokens", 300)
        temperature    = params.get("temperature", 0.7)

        # ── Build prompt ───────────────────────────────────────────────────────
        prompt = f"User: {message.strip()}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # ── Generate ───────────────────────────────────────────────────────────
        stopping_criteria = StoppingCriteriaList([
            StopOnTokens(self.stop_ids, self.newline_user_ids)
        ])

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        # ── Decode only the newly generated tokens ─────────────────────────────
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # ── Post-process: safety net for any markers that still leak through ───
        for marker in ["User:", "Assistant:", "|end_of_text|", "<end_of_turn>", "<b>", "</b>"]:
            if marker in response:
                response = response.split(marker)[0].strip()

        return [{"generated_text": response}]

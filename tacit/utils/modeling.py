"""Model and tokenizer loading, and trajectory rendering.

A trajectory is rendered as a chat through the backbone's own template: the profile and
the tool schemas form the system message, agent thought and action form assistant turns,
and each environment observation is a user turn.
"""
from __future__ import annotations

from ..data.schema import Trace
from .. import config as C


def trace_to_messages(trace: Trace, include_thought: bool = True,
                      rich_tools: bool = False) -> list[dict[str, str]]:
    """Role/content messages of a trajectory (``goal`` is not rendered)."""
    msgs: list[dict[str, str]] = []
    sys_parts = []
    if (trace.profile or "").strip():
        sys_parts.append(trace.profile.strip())
    if trace.tools:
        sys_parts.append(_tools_block(trace.tools, rich=rich_tools))
    if sys_parts:
        msgs.append({"role": "system", "content": "\n\n".join(sys_parts)})
    for t in trace.turns:
        if t.role == "user":
            msgs.append({"role": "user", "content": t.content or ""})
        elif t.role == "agent":
            parts = []
            if include_thought and t.thought:
                parts.append(f"Thought: {t.thought.strip()}")
            if t.action:
                parts.append(f"Action: {t.action.strip()}")
            body = "\n".join(parts) if parts else (t.content or "")
            msgs.append({"role": "assistant", "content": body})
        elif t.role == "environment":
            msgs.append({"role": "user", "content": f"[Observation] {t.content or ''}"})
    return _merge_consecutive(msgs)


def _tools_block(tools: list[dict], rich: bool = False) -> str:
    """Tool schemas as text. rich=True adds each parameter's type, default and
    description; otherwise parameters are listed by name."""
    lines = ["# Available tools"]
    for t in tools:
        name = t.get("name", "?")
        desc = (t.get("description", "") or "").strip()
        proto = t.get("prototype", "")
        params = t.get("parameters", {})
        head = f"- {name}: {desc}"
        if proto:
            head += f"\n  signature: {proto}"
        flat = {}
        if isinstance(params, dict):
            flat = params["properties"] if isinstance(params.get("properties"), dict) \
                else params
        if rich and flat:
            plines = []
            for p, spec in flat.items():
                if not isinstance(spec, dict):
                    continue
                ty = spec.get("type", "")
                bits = [b for b in (ty,) if b]
                if "default" in spec and spec["default"] is not None:
                    bits.append(f"default {spec['default']!r}")
                d = (spec.get("description") or "").strip()
                plines.append(f"  - {p}" + (f" ({', '.join(bits)})" if bits else "")
                              + (f": {d}" if d else ""))
            if plines:
                head += "\n  parameters:\n" + "\n".join(plines)
        elif flat:
            head += f"\n  parameters: {', '.join(flat.keys())}"
        lines.append(head)
    return "\n".join(lines)


def _merge_consecutive(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge runs of same-role messages, for templates that expect alternation."""
    out: list[dict[str, str]] = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1]["content"] = (out[-1]["content"] + "\n" + m["content"]).strip()
        else:
            out.append(dict(m))
    return out


def _fold_system(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Fold a leading system message into the first user turn, for templates without
    a system role."""
    if not msgs or msgs[0]["role"] != "system":
        return msgs
    sys = msgs[0]["content"]
    out, folded = [], False
    for m in msgs[1:]:
        if not folded and m["role"] == "user":
            out.append({"role": "user", "content": sys + "\n\n" + m["content"]})
            folded = True
        else:
            out.append(dict(m))
    return out if folded else [{"role": "user", "content": sys}] + out


def render(tokenizer, trace: Trace, include_thought: bool = True,
           rich_tools: bool = False) -> str:
    """The full chat string of a trajectory under the model's own template."""
    msgs = trace_to_messages(trace, include_thought=include_thought,
                             rich_tools=rich_tools)
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        return tokenizer.apply_chat_template(
            _fold_system(msgs), tokenize=False, add_generation_prompt=False)


def trace_to_plaintext(trace: Trace, include_tools: bool = True) -> str:
    """Flat readable serialisation of a trajectory."""
    parts: list[str] = []
    if (trace.profile or "").strip():
        parts.append(f"[System] {trace.profile.strip()}")
    if include_tools and trace.tools:
        parts.append(_tools_block(trace.tools))
    for t in trace.turns:
        if t.role == "user":
            parts.append(f"[User] {t.content or ''}")
        elif t.role == "agent":
            body = []
            if t.thought:
                body.append(f"Thought: {t.thought}")
            if t.action:
                body.append(f"Action: {t.action}")
            parts.append("[Agent] " + (" ".join(body) if body else (t.content or "")))
        elif t.role == "environment":
            parts.append(f"[Tool Result] {t.content or ''}")
    return "\n".join(parts)


def encode_truncated(tok, text: str, max_length: int):
    """Tokenise; over max_length, keep the first and last halves. Returns
    (ids, truncated)."""
    import torch
    ids = tok(text, return_tensors="pt").input_ids[0]
    if ids.shape[0] <= max_length:
        return ids, False
    head = max_length // 2
    return torch.cat([ids[:head], ids[-(max_length - head):]]), True


def load_tokenizer(model_key: str):
    from transformers import AutoTokenizer
    hf_id = C.MODELS.get(model_key, model_key)
    return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)


def load_model_and_tokenizer(model_key: str, dtype: str = "bfloat16", device: str = "cuda"):
    """Short name from config.MODELS, Hugging Face id, or local checkpoint directory."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = C.MODELS.get(model_key, model_key)
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=getattr(torch, dtype), trust_remote_code=True).to(device).eval()
    return model, tok

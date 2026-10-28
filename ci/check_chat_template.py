#!/usr/bin/env python3
"""Build-time check of the chat template the image ships at /opt/qwen-fixed.jinja.

Renders it the way vLLM does (transformers' sandboxed Jinja environment): thinking on and off, the
xhigh effort instruction, and a tool round trip with the arguments both as a JSON string (what
OpenAI clients send back) and as a mapping (what vLLM hands the template after parsing). A template
that does not render, or renders these differently, fails the image build instead of the first
chat request. Run in the final stage of the Dockerfile; no GPU needed.
"""
from transformers.utils.chat_template_utils import render_jinja_template
t = open("/opt/qwen-fixed.jinja").read()
assert 'template_version = "qwen3.8-froggeric-v22.5-sly1"' in t
tools = [{"type": "function", "function": {"name": "get_weather", "description": "w",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Wetter in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "call the tool",
         "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "content": '{"temp": 21}'}]
def render(m, **kw):
    out, _ = render_jinja_template(conversations=[m], tools=tools, chat_template=t, add_generation_prompt=True, **kw)
    return out[0]
on = render(msgs, reasoning_effort="medium")
off = render(msgs, enable_thinking=False)
assert on.endswith("<|im_start|>assistant\n<think>\n") and off.endswith("<think>\n\n</think>\n\n"), (on[-60:], off[-60:])
# stringified JSON arguments (OpenAI clients) render raw instead of crashing ...
assert '<function=get_weather>\n{"city": "Paris"}</function>' in on, on
# ... and a mapping (what vLLM hands over after parsing) renders as XML parameters
msgs[2]["tool_calls"][0]["function"]["arguments"] = {"city": "Paris"}
assert "<function=get_weather>\n<parameter=city>\nParis\n</parameter>" in render(msgs)
assert "<tool_response>\n{\"temp\": 21}" in on
# sly1: replayed reasoning of earlier assistant turns is dropped by default (an agent that sends its old
# thinking back locks into the repetition loops it contains); the empty think block stays, so the turn
# shape is the one measured. preserve_thinking / preserve_reasoning set explicitly = upstream behaviour.
assert "call the tool" not in on and "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>" in on, on
assert "call the tool" in render(msgs, reasoning_effort="medium", preserve_thinking=True)
assert "call the tool" in render(msgs, reasoning_effort="medium", preserve_reasoning=True)
# explicit false keeps upstream semantics: reasoning after the last user query survives
assert "call the tool" in render(msgs, reasoning_effort="medium", preserve_thinking=False)
# reasoning inlined in content (<think>...</think>answer) is dropped as well
inl = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>\nold plan\n</think>\n\nanswer"},
       {"role": "user", "content": "q2"}]
o = render(inl)
assert "old plan" not in o and "<think>\n\n</think>\n\nanswer" in o, o
# sly2: only the MOST RECENT assistant turn keeps its reasoning; older turns render an empty block
three = [{"role": "user", "content": "u1"},
         {"role": "assistant", "content": "a1", "reasoning_content": "REASON-ONE"},
         {"role": "user", "content": "u2"},
         {"role": "assistant", "content": "a2", "reasoning_content": "REASON-TWO"},
         {"role": "user", "content": "u3"}]
latest = render(three)
assert "REASON-TWO" in latest and "REASON-ONE" not in latest, latest
# ... and preserve_thinking stays the full-continuity opt-in
kept_opt_in = render(three, preserve_thinking=True)
assert "REASON-ONE" in kept_opt_in and "REASON-TWO" in kept_opt_in, kept_opt_in
# sly3: older turns keep only the HEAD of their reasoning (plan, decision); the tail is where the
# repetition lives. 0 reproduces sly2, a negative value keeps everything.
def _long(tag):
    return f"HEAD-{tag} plan\n\n" + ("filler sentence about the state. " * 60) + f"\n\nTAIL-{tag} conclusion"
turns3 = [{"role": "user", "content": "u1"},
          {"role": "assistant", "content": "a1", "reasoning_content": _long("ONE")},
          {"role": "user", "content": "u2"},
          {"role": "assistant", "content": "a2", "reasoning_content": _long("TWO")},
          {"role": "user", "content": "u3"}]
head = render(turns3)
assert "HEAD-ONE" in head and "TAIL-ONE" not in head, "older turn must keep the head only"
assert "HEAD-TWO" in head and "TAIL-TWO" in head, "newest turn must keep its full reasoning"
cut0 = render(turns3, old_reasoning_head_chars=0)
assert "HEAD-ONE" not in cut0 and "TAIL-TWO" in cut0, "0 must reproduce sly2"
kept = render(turns3, old_reasoning_head_chars=-1)
assert "TAIL-ONE" in kept, "a negative value must keep everything"
x = render([{"role": "user", "content": "hi"}], reasoning_effort="xhigh")
assert "Reasoning effort is set to xhigh" in x
print("chat template OK: qwen3.8-froggeric-v22.5-sly1 renders (thinking on/off, xhigh, tool round trip)")

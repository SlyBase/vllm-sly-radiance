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
assert 'template_version = "qwen3.8-froggeric-v22.5-sly2"' in t
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
# sly2: the tool-calling turn is the newest assistant turn, so it keeps its reasoning (older turns are
# checked below). preserve_thinking / preserve_reasoning set explicitly = upstream behaviour.
assert "<|im_start|>assistant\n<think>\ncall the tool\n</think>\n\n<tool_call>" in on, on
assert "call the tool" in render(msgs, reasoning_effort="medium", preserve_thinking=True)
assert "call the tool" in render(msgs, reasoning_effort="medium", preserve_reasoning=True)
# explicit false keeps upstream semantics: reasoning after the last user query survives
assert "call the tool" in render(msgs, reasoning_effort="medium", preserve_thinking=False)
# reasoning inlined in the content of an OLDER turn (<think>...</think>answer) is dropped as well
inl = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>\nold plan\n</think>\n\nanswer"},
       {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2", "reasoning_content": "new plan"},
       {"role": "user", "content": "q3"}]
o = render(inl)
assert "old plan" not in o and "<think>\n\n</think>\n\nanswer" in o and "new plan" in o, o
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
x = render([{"role": "user", "content": "hi"}], reasoning_effort="xhigh")
assert "Reasoning effort is set to xhigh" in x
print("chat template OK: qwen3.8-froggeric-v22.5-sly2 renders (thinking on/off, xhigh, tool round trip)")

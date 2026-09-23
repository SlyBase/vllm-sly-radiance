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
assert 'template_version = "qwen3.8-froggeric-v22.5"' in t
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
assert "<tool_response>\n{\"temp\": 21}" in on and "call the tool" in on
x = render([{"role": "user", "content": "hi"}], reasoning_effort="xhigh")
assert "Reasoning effort is set to xhigh" in x
print("chat template OK: qwen3.8-froggeric-v22.5 renders (thinking on/off, xhigh, tool round trip)")

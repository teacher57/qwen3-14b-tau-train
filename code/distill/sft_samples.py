"""Turn teacher conversations into per-turn SFT samples laid out exactly as at inference.

For every assistant message i of a conversation:
  prompt     = chat template of messages[:i] (+ tools) with add_generation_prompt=True   (what vLLM feeds the model)
  completion = whatever the template adds for message i  ("<think>...</think>" + text + tool calls + <|im_end|>)
Loss is taken on the completion only. Because each turn is its own sample, every turn's reasoning is supervised, even
though Qwen3's template strips the thinking of earlier turns from the history.
"""


def build_samples(messages, tools, tok, max_tokens=20000):
    samples, skipped = [], {"template_mismatch": 0, "too_long": 0}
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prompt_text = tok.apply_chat_template(messages[:i], tools=tools, tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(messages[: i + 1], tools=tools, tokenize=False)
        if not full_text.startswith(prompt_text):
            skipped["template_mismatch"] += 1
            continue
        completion = full_text[len(prompt_text):]
        if completion.endswith("<|im_end|>\n"):
            completion = completion[:-1]
        p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > max_tokens:
            skipped["too_long"] += 1
            continue
        samples.append({"prompt_ids": p_ids, "completion_ids": c_ids, "turn": i})
    return samples, skipped

"""Like sft_samples.build_samples, but only the assistant turns listed in `supervise` become samples (all others stay as unsupervised context)."""
def build_samples_idx(messages, tools, tok, idxs, max_tokens=20000):
    samples, skipped = [], {"template_mismatch": 0, "too_long": 0}
    for i in idxs:
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

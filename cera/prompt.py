SYSTEM = (
    "You are a chest CT report evaluator. Compare the CANDIDATE report against the "
    "REFERENCE report and list every clinically meaningful error in the candidate. "
    "Return a JSON list; return [] if the candidate contains no errors.\n"
    "category: false_finding | omission | location | severity | comparison\n"
    "region: Lungs & Airways | Cardiovascular | Mediastinum & Hila | Pleura | "
    "Upper Abdomen | Bones / Spine | Chest Wall | Lower Neck | Others\n"
    "criticality: Critical | Minor"
)

USER = "REFERENCE:\n{ref}\n\nCANDIDATE:\n{cand}"


def messages(ref, cand):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(ref=ref, cand=cand)}]


def render(tok, msgs):
    # Qwen3.5-4B leaves <think> open by default, which overrides the adapter.
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# Train / inference consistency

The students were trained on single decisions exported from teacher runs of H\*. A mismatch between how a request is
built at training time and at evaluation time does not raise an error; it only lowers the success rate. Check these
points before changing the serving stack, the chat template or the harness:

1. **No reasoning in the history.** Earlier assistant turns carry only the text after `</think>` (the tool call). The
   harness strips the reasoning itself; do not rely on the chat template for it.
2. **Image history.** Only the most recent K user turns keep their images: K = 3 with one image per turn for Guava /
   LIBERO-PRO (and the zero-shot drivers built on them), K = 2 with three images per turn (head + both wrists) for RoboTwin. Serve
   the model with at least 8 images per request (`--limit-mm-per-prompt '{"image":8}'` in vLLM).
3. **Thinking.** Requests set `chat_template_kwargs.enable_thinking` (`--thinking 1`, the reported setting). Qwen3.5
   outputs end the reasoning with `</think>` and do not always open it with `<think>`; parsing never depends on the
   opening tag.
4. **One sample = one decision**, not one episode (see docs/training.md).
5. **`--gen` flags are part of the prompt.** `--gen default` appends a paragraph to the SYSTEM prompt and adds result
   codes / fields. All reported student and teacher results use `--gen default`.
6. **Chat template.** The student checkpoints must render prompts exactly like the base Qwen3.5 tokenizer
   (`serving/gpu/check_chat_template.py`).
7. **Served model name.** The harness sends `"model": "qwen"` unless `--model` is given; serve with
   `--served-model-name qwen` or pass `--model`.
8. **RoboTwin measurement noise.** The measured gripper positions in the `Arms:` line can differ by 0.1 cm between runs
   and renders are not bit-exact; compare requests with that line masked (tests/equiv_train.py compares text with images
   replaced).

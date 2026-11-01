#!/usr/bin/env python
"""Check that a fine-tuned checkpoint renders prompts exactly like its base model.

The students are trained with the base model's chat template; a checkpoint whose saved template differs
(e.g. a template that keeps <think> in history turns, or a different generation prompt) silently shifts the
input distribution at inference time.  This renders one or more conversations with both tokenizers
(add_generation_prompt=True, enable_thinking=True) and requires byte-identical output.

usage: python check_chat_template.py BASE_MODEL CHECKPOINT [--data sft.jsonl] [--n 1]
  BASE_MODEL  e.g. Qwen/Qwen3.5-4B (hub id or local dir)
  CHECKPOINT  the student directory to check
  --data      an SFT jsonl in the train/ format; the last assistant turn of each row is dropped and the rest is
              rendered.  Without --data a small built-in multi-turn conversation is used.
Exit code 0 = IDENTICAL, 1 = DIFFERENT.
"""
import argparse
import json
import sys

from transformers import AutoTokenizer

BUILTIN = [
    {"role": "system", "content": "You control a robot arm through tools."},
    {"role": "user", "content": "<image>Task: pick up the apple."},
    {"role": "assistant", "content": "\n\n<tool_call>\n{\"name\": \"locate\", \"arguments\": {\"objects\": [\"apple\"]}}\n</tool_call>"},
    {"role": "user", "content": "<image>TOOL RESULT: {\"apple\": \"found\"}"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    tb = AutoTokenizer.from_pretrained(a.base)
    tc = AutoTokenizer.from_pretrained(a.checkpoint)
    convs = []
    if a.data:
        with open(a.data) as f:
            for line in f:
                x = json.loads(line)
                convs.append([{"role": m["role"], "content": m["content"]} for m in x["messages"][:-1]])
                if len(convs) >= a.n:
                    break
    else:
        convs = [BUILTIN]
    same = True
    for i, msgs in enumerate(convs):
        ra = tb.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        rb = tc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ok = ra == rb
        same &= ok
        print(f"[{i}] template render vs base (enable_thinking=true): {'IDENTICAL' if ok else 'DIFFERENT'}; ends with {rb[-30:]!r}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()

"""Toy harness H0: two tools wrapping the environment; the teacher chooses the next tool call each turn.
Deliberate weakness for the example: every grasp approaches from the top, which rarely works on tall objects."""
import json

SYSTEM = ("You control one robot arm. Reply with exactly one JSON tool call per turn: "
          '{"tool": "grasp", "object": X} | {"tool": "place", "object": X, "target": Y} | {"tool": "done"}')


def tool_grasp(env, obj, log):
    ok = env.grasp(obj, approach="top")
    log.append(f"grasp {obj} approach=top -> {'OK' if ok else 'MISSED'}")
    return ok


def tool_place(env, obj, target, log):
    ok = env.place(obj, target)
    log.append(f"place {obj} -> {target}: {'OK' if ok else 'FAILED'}")
    return ok


def run(env, teacher, log, max_calls=6):
    msgs = [{"role": "system", "content": SYSTEM}]
    for _ in range(max_calls):
        msgs.append({"role": "user", "content": env.observe()})
        reply = teacher.chat(msgs)
        msgs.append({"role": "assistant", "content": reply})
        log.append(f"teacher: {reply}")
        call = json.loads(reply)
        if call["tool"] == "done":
            return
        if call["tool"] == "grasp":
            env.last = "grasp " + ("OK" if tool_grasp(env, call["object"], log) else "MISSED")
        elif call["tool"] == "place":
            env.last = "place " + ("OK" if tool_place(env, call["object"], call["target"], log) else "FAILED")

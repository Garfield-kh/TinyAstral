"""Apply a unified diff to a sandbox copy of the parent harness, and lint it (stdlib only)."""
import re
import shutil

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _files(diff):
    """[(path, [hunk lines...]), ...] from a unified diff (paths taken from +++ lines, a/ b/ prefixes stripped)."""
    out, cur = [], None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].split("\t")[0].strip()
            path = path[2:] if path[:2] in ("a/", "b/") else path
            cur = (path, [])
            out.append(cur)
        elif line.startswith("--- ") or line.startswith("diff ") or line.startswith("index "):
            continue
        elif cur is not None:
            cur[1].append(line)
    return out


def _apply_file(old, hunks):
    src, res, pos = old, [], 0
    i = 0
    while i < len(hunks):
        m = HUNK.match(hunks[i])
        if not m:
            i += 1
            continue
        body = []
        i += 1
        while i < len(hunks) and not HUNK.match(hunks[i]):
            body.append(hunks[i])
            i += 1
        before = [l[1:] for l in body if l[:1] in (" ", "-")]
        after = [l[1:] for l in body if l[:1] in (" ", "+")]
        start = int(m.group(1)) - 1 if before else int(m.group(1))
        hit = next((s for d in range(len(src) + 1) for s in (start + d, start - d)
                    if 0 <= s <= len(src) - len(before) and s >= pos and src[s:s + len(before)] == before), None)
        if hit is None:
            raise ValueError(f"hunk does not apply near line {start + 1}")
        res += src[pos:hit] + after
        pos = hit + len(before)
    return res + src[pos:]


def apply(parent_dir, cand_dir, diffs):
    """Copy parent_dir -> cand_dir and apply each diff in order there.  The parent is never modified."""
    shutil.copytree(parent_dir, cand_dir)
    for diff in ([diffs] if isinstance(diffs, str) else diffs):
        for path, hunks in _files(diff):
            f = cand_dir / path
            old = f.read_text().splitlines() if f.exists() else []
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("\n".join(_apply_file(old, hunks)) + "\n")


def lint(diff, tasks, markers, targets=(), frozen=(), memory_dir=None):
    """Problems that make a candidate inadmissible before any episode is run.
    frozen: harness paths that must stay byte-identical (e.g. the system prompt and existing tool descriptions).
    memory_dir: task-memory notes live there as <task>.<ext>; only the target tasks' notes may change."""
    probs = [] if markers else ["no trigger marker declared"]
    code = []
    for path, hunks in _files(diff):
        added = [l[1:] for l in hunks if l.startswith("+")]
        if any(path == f or path.startswith(f.rstrip("/") + "/") for f in frozen):
            probs.append(f"{path} is frozen and must stay byte-identical to the parent")
        if memory_dir and path.startswith(memory_dir.rstrip("/") + "/"):
            stem = path.rsplit("/", 1)[-1].split(".")[0]
            if stem not in targets:
                probs.append(f"memory note {path} is not for a target task")
        else:
            code += added
    added = "\n".join(code)
    probs += [f"marker {m!r} is never written by the added code" for m in markers if m not in added]
    probs += [f"task name {t!r} appears in the added code (edits must be general)" for t in tasks
              if re.search(r"\b" + re.escape(t) + r"\b", added)]
    return probs

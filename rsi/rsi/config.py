"""Pre-registered rules.  Written by humans before the loop starts; hashed into every ledger record."""
import hashlib
import json
import pathlib

DEFAULTS = {
    "budget": {"max_rounds": 5, "max_candidates_per_round": 3, "max_episodes": 100000, "workers": 8},
    "dev": {"pairs_per_task": 5, "max_failures_shown": 30},
    "gate": {"bootstrap_reps": 4000, "bootstrap_seed": 7, "alpha": 0.025, "veto_untriggered_pp": -2.0,
             "interim_fraction": 0.5, "interim_planned_gain_pp": 3.0, "interim_min_power": 0.10,
             "screen_pairs_per_task": 4, "screen_min_upper_pp": 2.0},
}


def _scalar(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return [_scalar(x) for x in s[1:-1].split(",") if x.strip()]
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "none", "~", ""):
        return None
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return s.strip("'\"")


def _mini_yaml(text):
    """Nested mappings by indentation, scalars, inline [a, b] lists, # comments.  Enough for rules.yaml."""
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack[-1][0] >= ind:
            stack.pop()
        if val.strip():
            stack[-1][1][key] = _scalar(val)
        else:
            stack[-1][1][key] = {}
            stack.append((ind, stack[-1][1][key]))
    return root


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_rules(path):
    p = pathlib.Path(path)
    raw = p.read_text()
    if p.suffix == ".json":
        cfg = json.loads(raw)
    else:
        try:
            import yaml
            cfg = yaml.safe_load(raw)
        except ImportError:
            cfg = _mini_yaml(raw)
    cfg = _merge(DEFAULTS, cfg)
    cfg["_path"], cfg["_dir"] = str(p), str(p.parent)
    cfg["_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    for k in ("teacher", "tasks", "seeds", "runner", "harness", "workdir"):
        assert k in cfg, f"rules file must define `{k}`"
    return cfg


def resolve(cfg, rel):
    """Paths in the rules file are relative to the rules file."""
    q = pathlib.Path(rel)
    return q if q.is_absolute() else pathlib.Path(cfg["_dir"]) / q

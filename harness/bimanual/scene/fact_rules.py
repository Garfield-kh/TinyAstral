"""Conservative contradiction checks against the current observation only."""
import re

FACT_RULES = '''
FACTUALITY RULES (apply to seen, reported, assessment and memory_delta):
- Use exact canonical object names, especially numbered objects (block1, block2, block3).
- Do not use colour descriptions for numbered blocks unless the identity is directly established; omit them instead.
- A view claim must refer to an object actually visible in THAT view, not merely in overview or measured state.
- Avoid front/back/rear spatial descriptions; report measured coordinates without interpreting the y sign.
- Never assert a button has or has not been activated without explicit feedback/checker evidence; state uncertainty.
- A failed approach does not mean the gripper closed. Check the current command and the actual failure stage.
- Do not call the floor dustbin a sheet or infer where falling trash landed from its projected image position.
- Do not infer a mug is upright from its bounding-box dimensions; orientation needs direct evidence.
- Omit toward/away/forward directional arithmetic. Numeric coordinates alone are enough evidence.
- Do not claim an earlier gripper closure based only on a failed approach or a later locate result.
- Assessment evaluates ONLY what has happened. Put the desired condition in next, without numeric movement amounts.
- Use canonical object names. Do not guess colour, identity or relative position from an ambiguous image.
- Put measured gripper/holding facts in reported. A closed gripper does not imply an object is held;
  contact, commanded opening, physical opening and holding are different facts.
- Current measured holding overrides earlier successful feedback. If the current state says holding nothing,
  do not claim a secure hold; mention uncertainty if the visual evidence conflicts.
- Half-closed is neither fully open nor fully closed. Always name the arm in a gripper claim.
- In two-hand tasks, check EACH arm separately. Do not infer handover/release from a one-arm movement.
- Resting, tipped, floating and suspended are different. If support is unclear, omit the claim or mark uncertainty.
- Assessment must be supported by current evidence, not by the intended effect of a previous action.
- Next is a desired observable condition, not an imperative or a tool call; no action parameters or tool names,
  including inflections (grasps, grasping, placing, rotating, pressing).
'''


def audit(cot, state, objects):
    bad = []
    names = list(objects)
    aliases = {n: {n.lower(), n.lower().replace('_', ' ')} for n in names}
    for n in names:
        for word in re.findall(r'[a-z]+', n.lower()):
            if len(word) >= 3 and sum(word in x.lower() for x in names) == 1:
                aliases[n].add(word)
        for short in ('pot', 'mug', 'clock', 'roller'):
            if short in n and sum(short in x for x in names) == 1:
                aliases[n].add(short)
    for field in ('seen', 'reported', 'assessment', 'memory_delta'):
        values = cot.get(field) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            s = str(value)
            low = re.sub(r'^(overview|left wrist|right wrist|top):\s*', '', s.lower())
            if re.search(r'\b(will|would|should|might|may|perhaps|unclear|uncertain)\b', low):
                continue
            obs = [n for n in names if any(re.search(r'\b' + re.escape(a) + r'\b', low) for a in aliases[n])]
            arms = set(re.findall(r'\b(left|right)\b', low))
            reason = []
            if not arms and re.search(r'\b(open|opened|closed|shut)\s+gripper\b|\bgripper\b.{0,15}\b(open|closed)\b', low):
                reason.append('gripper claim must explicitly identify its arm')
            for side in ('left', 'right'):
                if state['held'].get(side) and re.search(r"out of (?:the )?" + side + r" (?:hand|gripper)(?:'s)? (?:grip|hold)", low):
                    reason.append(f"{side} still holds {state['held'][side]}")
            if len(obs) == 1:
                color = (state.get('colors') or {}).get(obs[0], {}).get('name')
                words = set(re.findall(r'\b(red|orange|yellow|green|blue|purple|pink|magenta|cyan|brown|black|white|gray|grey|cream)\b', low))
                allowed = {'gray': {'gray','grey','cream'}, 'pink': {'pink','magenta','purple'},
                           'purple': {'purple','magenta','pink'}}.get(color, {color})
                if color and words and not words.intersection(allowed):
                    reason.append(f"colour claim conflicts with measured {obs[0]} colour={color}")
            if len(arms) == 1:
                arm = next(iter(arms)); g = state['arms'][arm]['gripper_cmd']; held = state['held'].get(arm)
                open_claim = re.search(r'\bopen(?:ed)?\s+(?:gripper|fingers)\b|\b(?:gripper|fingers|arm)\b.{0,25}\bopen(?:ed)?\b', low)
                closed_claim = re.search(r'\b(?:closed|shut)\s+(?:gripper|fingers)\b|\b(?:gripper|fingers|arm)\b.{0,25}\b(?:closed|shut)\b', low)
                neg = re.search(r'\b(not|no longer|never)\b', low)
                if not neg and not re.search(r'\bhalf|partially|partly|pre.?shap', low):
                    if open_claim and g <= .8: reason.append(f'{arm} gripper is not fully open (cmd={g})')
                    if closed_claim and g >= .2: reason.append(f'{arm} gripper is not fully closed (cmd={g})')
                holding = re.search(r'\b(holds|holding|grips|gripping|grasping|grasped|held|secured)\b', low)
                if holding and len(obs) == 1 and not re.search(r'\b(not|nothing|no longer|without|failed)\b', low):
                    if held != obs[0]: reason.append(f'{arm} holding={held}, not {obs[0]}')
                if held and re.search(r'\b(empty|holds nothing|holding nothing|released|let go|out of .{0,20}grip)\b', low) and not neg:
                    reason.append(f'{arm} still holds {held}')
            if len(obs) == 1 and not re.search(r'\b(not|no longer|never)\b', low):
                n = obs[0]; z = state['objects'][n]['lo'][2]
                if re.search(r'\b(floating|in the air|off the table|suspended)\b', low) and z < .8:
                    reason.append(f'{n} bottom z={z} does not support an airborne claim')
                if re.search(r'\b(rests|resting|lies|lying|sits|sitting|stands|standing) on the table\b', low) and z > 3:
                    reason.append(f'{n} bottom z={z} contradicts table support')
            for r in reason:
                bad.append(f"{field}: '{s}' ({r})")
    bad.extend(strict_evidence_audit(cot, state, objects))
    return bad


def strict_evidence_audit(cot, state, objects):
    bad = []
    numbered = [n for n in objects if re.fullmatch(r"block[0-9]+", n)]
    for field in ('seen', 'reported', 'assessment', 'memory_delta', 'next'):
        values = cot.get(field) or []
        if isinstance(values, str): values = [values]
        for value in values:
            s = str(value); low = s.lower(); why = []
            if field != 'next' and re.search(r"\b(front|rear)\b|\bback of\b|\bcm back\b|\bpointing toward the back\b", low):
                why.append('omit ambiguous front/back/rear; use measured coordinates without interpreting y')
            if field in ('assessment','reported','memory_delta') and re.search(r"button.{0,40}(?:has|been|is).{0,15}(?:pressed|activated)|button has not", low):
                why.append('button activation is not established by motion/holding feedback; express uncertainty')
            if field == 'seen':
                m = re.match(r"(overview|left wrist|right wrist):", low)
                if m:
                    cam = {'overview':'agent','left wrist':'left','right wrist':'right'}[m[1]]
                    vis = state.get('visible', {}).get(cam, {})
                    if not re.search(r"not (?:in|visible|yet)|not.*view|does not.*show|too.*(?:blurry|close)|unclear", low):
                        for n in objects:
                            if re.search(r"\b"+re.escape(n.replace('_',' '))+r"\b",low.replace('_',' ')) and int(vis.get(n,0)) == 0:
                                why.append(f'{n} has no visible pixels in {cam}; omit this view claim')
            if numbered and re.search(r"\b(?:purple|pink|white|blue|red|green|yellow|orange) (?:cube|block)\b", low):
                why.append('numbered block colour alias is ambiguous; use its exact canonical name without colour')
            if re.search(r"\bgrasp closed\b",low):
                sides = re.findall(r"\b(left|right)\b",low)
                if len(set(sides)) == 1 and state['arms'][sides[0]]['gripper_cmd'] > .8:
                    why.append('current gripper command is open; failed approach does not establish a close')
            if field == 'next' and re.search(r"^(bring|find|move|open|close|lift|release|set|pick|first find)\b|; first find\b", low):
                why.append('next must be a desired observable state, not an imperative')
            if field != 'next':
                if 'dustbin' in objects and re.search(r"\b(sheet|canvas|cloth)\b", low):
                    why.append('do not identify the floor dustbin as a sheet or infer a landing surface from projection')
                if 'mug' in objects and re.search(r"\bupright\b|long axis vertical",low):
                    why.append('mug orientation is not established by its bounding-box dimensions; omit upright inference')
                if re.search(r"\b(?:toward|towards|away from) (?:the )?robot\b|\bcm forward\b",low):
                    why.append('omit directional interpretation of coordinates; report measured positions only')
                if re.search(r"(?:gripper|grasp|arm).{0,18}closed",low):
                    sides=set(re.findall(r"\b(left|right)\b",low))
                    if len(sides)==1 and state['arms'][next(iter(sides))]['gripper_cmd']>.8:
                        why.append('gripper currently commanded open; do not infer historical closure from failed approach')
                if field=='assessment' and re.search(r"(?:next step|remaining step|remaining part).{0,80}(?:move|close|open|press|grasp|place|release)",low):
                    why.append('assessment must evaluate completed events, not specify the next action')
            if field=='next' and re.search(r"(?:raise|move|shift|lower|lift|turn|rotate).{0,30}\b[0-9]+(?:\.[0-9]+)?\s*(?:cm|degree)",low) and 'at least' not in low:
                why.append('next must omit numeric action amounts; use a qualitative desired state')
            for reason in why: bad.append(f"{field}: '{s}' ({reason})")
    return bad

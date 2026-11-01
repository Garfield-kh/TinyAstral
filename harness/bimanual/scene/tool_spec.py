"""Shared documentation for the corrected tool adapter (synthesis and all ICL arms)."""
TOOL_DOC = '''

# Corrected tool interface (applies equally with and without an example)
- move(arm, dx=0, dy=0, dz=0, rx=0, ry=0, rz=0): move fingertips in cm and optionally
  rotate in degrees in ONE planned motion. Rotation is world-frame extrinsic x, then y, then z,
  applied about the fingertips. The robot computes the corresponding wrist target.
- grasp(arm, object, part=null, lift_cm=null, contact=null, approach_cm=null, preshape=null):
  contact optionally selects an object's contact-point ID (or list); it overrides the part heuristic.
  approach_cm is the pre-grasp distance (default 10 cm). preshape is opening in [0,1] retained
  during approach. lift_cm=0 means no lift. Do not invent a contact ID when it is not known.
- place additionally accepts target="at:x,y,z" in table-frame cm for an explicit placement point.
  It tries functional-point placement, then free object-pose placement if unreachable. This differs
  from a semantic target; choose coordinates from the currently observed scene, not from an example.
- Examples may contain precise motion parameters for another scene. Re-evaluate all parameters
  against the current observation; never treat an example's success as evidence of current success.
'''

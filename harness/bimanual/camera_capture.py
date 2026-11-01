"""RGBD from exactly native head and wrist cameras; no extra-view fallback."""
import numpy as np

NAMES = ('head_camera', 'left_camera', 'right_camera')

def capture(rd):
    rd.env._update_render()
    C = rd.env.cameras
    static = list(getattr(C, 'static_camera_list', []))
    labels = list(getattr(C, 'static_camera_name', []))
    if len(static) != len(labels) or labels.count('head_camera') != 1:
        raise ValueError('native head camera must have exactly one named static mapping')
    if not getattr(C, 'collect_head_camera', False) or not getattr(C, 'collect_wrist_camera', False):
        raise ValueError('native head and both wrist cameras must be enabled')
    cameras = {'head_camera': static[labels.index('head_camera')]}
    cameras.update({name: getattr(C, name) for name in NAMES[1:]})
    views = []
    for name in NAMES:
        cam = cameras[name]
        cam.take_picture()
        rgba = np.asarray(cam.get_picture('Color'))
        pos = np.asarray(cam.get_picture('Position'))
        M = np.asarray(cam.get_model_matrix())
        if M.shape != (4, 4) or not np.isfinite(M).all():
            raise ValueError('invalid native camera matrix')
        if rgba.shape[:2] != pos.shape[:2] or rgba.ndim != 3 or pos.ndim != 3:
            raise ValueError('unaligned native RGB and depth')
        p = pos[..., :3]
        xyz = p @ M[:3, :3].T + M[:3, 3]
        valid = (pos[..., 2] < -1e-4) & np.isfinite(xyz).all(axis=-1)
        views.append({'name': name, 'rgb': (np.clip(rgba[..., :3], 0, 1)*255).astype(np.uint8),
                      'xyz': xyz, 'valid': valid})
    return views

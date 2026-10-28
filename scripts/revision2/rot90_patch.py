# -*- coding: utf-8 -*-
"""Online random 90-degree rotation for Ultralytics YOLODataset (IR2605 2nd revision, R1-5).

Applied per loaded image, right after load_image() and BEFORE mosaic, i.e. at the same point
where a pre-generated rotated copy would enter the pipeline as a mosaic tile.
Active only when the environment variable WELD_ROT90=1 and only for training datasets
(dataset.augment is True); validation is untouched. The env var is inherited by the
Windows spawn workers, and this module must be imported at the top of the launching script
so that the patch is re-applied when a worker re-imports __main__.

Box rule (normalized xyxy, one counter-clockwise quarter turn, as np.rot90(img, 1)):
    (x1, y1, x2, y2) -> (y1, 1 - x2, y2, 1 - x1)
Images in final0_clean are all 640 x 640, so shapes do not change; non-square input is refused.

Debug: WELD_ROT90_LOG=<dir> appends each drawn k to <dir>/k_<pid>.txt (verification only).
"""
import os
import random

import numpy as np
from ultralytics.data.dataset import YOLODataset

_ORIG = YOLODataset.get_image_and_label


def rot90_label(label, k):
    """Rotate label['img'] and label['instances'] by k quarter turns counter-clockwise."""
    k %= 4
    if k == 0:
        return label
    img = label["img"]
    h, w = img.shape[:2]
    if h != w:
        raise ValueError(f"rot90_patch expects square images, got {h}x{w}")
    label["img"] = np.ascontiguousarray(np.rot90(img, k))
    ins = label["instances"]
    fmt = ins._bboxes.format
    assert ins.normalized, "rot90_patch expects normalized boxes at this stage"
    ins.convert_bbox("xyxy")
    b = ins.bboxes
    for _ in range(k):
        x1, y1, x2, y2 = b[:, 0].copy(), b[:, 1].copy(), b[:, 2].copy(), b[:, 3].copy()
        b[:, 0], b[:, 1], b[:, 2], b[:, 3] = y1, 1.0 - x2, y2, 1.0 - x1
    if ins.segments is not None and len(ins.segments):
        for _ in range(k):
            x = ins.segments[..., 0].copy()
            ins.segments[..., 0] = ins.segments[..., 1]
            ins.segments[..., 1] = 1.0 - x
    ins.convert_bbox(fmt)
    return label


def _patched(self, index):
    label = _ORIG(self, index)
    if os.environ.get("WELD_ROT90") == "1" and getattr(self, "augment", False):
        k = random.randrange(4)
        log = os.environ.get("WELD_ROT90_LOG")
        if log:
            with open(os.path.join(log, f"k_{os.getpid()}.txt"), "a") as f:
                f.write(str(k))
        label = rot90_label(label, k)
    return label


YOLODataset.get_image_and_label = _patched

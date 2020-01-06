import numpy as np
import torch


def dice(seg_fixed, seg_moving):
    seg_fixed, seg_moving = seg_fixed.to(torch.short), seg_moving.to(torch.short)

    no_classes = torch.max(seg_fixed)
    scores = np.zeros(no_classes)

    for class_idx in range(no_classes):
        numerator = 2.0 * ((seg_fixed == (class_idx + 1)) * (seg_moving == (class_idx + 1))).sum()
        denominator = float((seg_fixed == (class_idx + 1)).sum() + (seg_moving == (class_idx + 1)).sum())

        dsc = numerator / denominator
        scores[class_idx] = dsc.item()

    return scores

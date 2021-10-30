import numpy as np
from scipy.ndimage import distance_transform_edt


def visualize_masks(mask, mask_pred):
    m = np.ones((256, 256, 3))
    m[np.logical_and(mask, mask_pred)] = np.array([0.1, 0.5, 0.1])
    m[np.logical_and(mask, np.logical_not(mask_pred))] = np.array([1, 0, 0])
    m[np.logical_and(np.logical_not(mask), mask_pred)] = np.array([0, 0, 1])
    return m


def compute_distance_transform(mask):
    dist_out = distance_transform_edt(1 - mask)
    dist_out = 2 * dist_out / max(mask.shape)
    return dist_out


def rle_to_binary_mask(rle):
    """
    rle should be coco format: {"counts": [], "size": []}
    """
    if isinstance(rle, list):
        return np.stack([rle_to_binary_mask(r) for r in rle])
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = list(map(int, counts.split(" ")))
    mask = np.zeros(np.prod(rle["size"]), dtype=bool)
    running_length = 0
    for start, length in zip(counts[::2], counts[1::2]):
        running_length += start
        mask[running_length : running_length + length] = 1
        running_length += length
    return mask.reshape(rle["size"], order="F")


def binary_mask_to_rle(binary_mask):
    counts = []
    last_elem = 0
    running_length = 0
    for elem in binary_mask.ravel(order="F"):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1
    counts.append(running_length)
    rle = {"counts": " ".join(map(str, counts)), "size": list(binary_mask.shape)}
    return rle

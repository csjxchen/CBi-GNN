import numpy as np


def extract_anno(image_shape, re, class_names):
    box2d = re['bbox']
    box3d = re['box3d_camera']
    labels = re['label_preds']
    scores = re['scores']
    alphas = re['alphas']
    anno = kitti.get_start_result_anno()
    num_example = 0
    for bbox2d, bbox3d, label, score, alpha in zip(box2d, box3d, labels, scores, alphas):
        if bbox2d[0] > image_shape[1] or bbox2d[1] > image_shape[0]:
            continue
        if bbox2d[2] < 0 or bbox2d[3] < 0:
            continue
        bbox2d[2:] = np.minimum(bbox2d[2:], image_shape[::-1])
        bbox2d[:2] = np.maximum(bbox2d[:2], [0, 0])
        anno["name"].append(class_names[int(label)])
        anno["truncated"].append(0.0)
        anno["occluded"].append(0)
        # anno["alpha"].append(-10)
        anno["alpha"].append(alpha)
        anno["bbox"].append(bbox2d)
        # anno["dimensions"].append(np.array([-1,-1,-1]))
        anno["dimensions"].append(bbox3d[[3, 4, 5]])
        # anno["location"].append(np.array([-1000,-1000,-1000]))
        anno["location"].append(bbox3d[:3])
        # anno["rotation_y"].append(-10)
        anno["rotation_y"].append(bbox3d[6])
        anno["score"].append(score)
        num_example += 1
    return anno, num_example

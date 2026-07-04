import numpy as np

from perception.segmentation import estimate_ground_depth, segment_objects


def test_estimate_ground_depth_is_dominant_value():
    depth = np.full((20, 20), 0.5)
    depth[2:6, 2:6] = 0.4
    assert estimate_ground_depth(depth) == 0.5


def test_segment_objects_finds_single_cluster():
    depth = np.full((20, 20), 0.5)
    depth[2:6, 2:6] = 0.4
    labeled, valid_ids = segment_objects(depth, ground_margin=0.01, min_cluster_pixels=10)
    assert len(valid_ids) == 1
    cluster_mask = labeled == valid_ids[0]
    assert np.count_nonzero(cluster_mask) == 16
    assert cluster_mask[2:6, 2:6].all()
    assert not cluster_mask[0:2, :].any()


def test_segment_objects_separates_two_clusters():
    depth = np.full((30, 30), 0.5)
    depth[2:6, 2:6] = 0.4
    depth[20:24, 20:24] = 0.4
    _, valid_ids = segment_objects(depth, ground_margin=0.01, min_cluster_pixels=10)
    assert len(valid_ids) == 2


def test_segment_objects_filters_small_noise_clusters():
    depth = np.full((20, 20), 0.5)
    depth[2:6, 2:6] = 0.4
    depth[10, 10] = 0.4
    _, valid_ids = segment_objects(depth, ground_margin=0.01, min_cluster_pixels=10)
    assert len(valid_ids) == 1

# config.py

cfg = {
    'name': 'FaceBoxes',
    'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'min_dim': 512,
    'steps': [16, 32, 64],
    'min_sizes': [[16, 32], [64], [128]],
    'aspect_ratios': [[1], [1], [1]],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0
}

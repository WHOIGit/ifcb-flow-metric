import yaml
from typing import Dict, Any


def load_feature_config(config_path: str) -> Dict[str, Any]:
    """Load feature configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_feature_config() -> Dict[str, Any]:
    """Get default feature configuration with all features enabled."""
    return {
        'spatial_stats': {
            'mean_x': True,
            'mean_y': True,
            'std_x': True,
            'std_y': True,
            'median_x': True,
            'median_y': True,
            'iqr_x': True,
            'iqr_y': True,
        },
        'distribution_shape': {
            'ratio_spread': True,
            'core_fraction': True,
        },
        'clipping_detection': {
            'duplicate_fraction': True,
            'max_duplicate_fraction': True,
        },
        'histogram_uniformity': {
            'cv_x': True,
            'cv_y': True,
        },
        'statistical_moments': {
            'skew_x': True,
            'skew_y': True,
            'kurt_x': True,
            'kurt_y': True,
        },
        'pca_orientation': {
            'angle': True,
            'eigen_ratio': True,
        },
        'edge_features': {
            'left_edge_fraction': True,
            'right_edge_fraction': True,
            'top_edge_fraction': True,
            'bottom_edge_fraction': True,
            'total_edge_fraction': True,
        },
        'temporal': {
            't_y_var': True,
        }
    }


def get_enabled_features(config: Dict[str, Any]) -> Dict[str, bool]:
    """Flatten the hierarchical config into a single dict of feature_name: enabled."""
    enabled = {}
    for category, features in config.items():
        if isinstance(features, dict):
            for feature_name, enabled_flag in features.items():
                enabled[feature_name] = enabled_flag
    return enabled
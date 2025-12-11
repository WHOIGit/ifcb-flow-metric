"""Configuration files and utilities."""

import importlib.resources
from pathlib import Path


def get_default_config_path():
    """Get path to default feature_config.yaml bundled with the package.

    Returns:
        Path or str: Path to the default feature_config.yaml file
    """
    # For Python 3.11+
    if hasattr(importlib.resources, 'files'):
        ref = importlib.resources.files('ifcb_flow_metric.config')
        return str(ref / 'feature_config.yaml')
    else:
        # Fallback for older Python versions
        import pkg_resources
        return pkg_resources.resource_filename('ifcb_flow_metric.config', 'feature_config.yaml')


__all__ = ["get_default_config_path"]

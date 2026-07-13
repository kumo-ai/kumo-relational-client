import sys

__version__ = '0.1.0'

# Preserve upstream absolute imports inside the vendored package.
sys.modules['kumopql'] = sys.modules[__name__]

__all__ = [
    '__version__',
]

import os

from setuptools import setup

_KUMO_API_VER = '0.92.0'
_KUMO_API_MAJOR = int(_KUMO_API_VER.split('.')[0])

if kumo_api_path := os.getenv('KUMO_API_PATH'):
    kumo_api = f'kumo-api @ file://{os.path.abspath(kumo_api_path)}'
elif os.getenv('KUMO_API_GIT'):
    kumo_api = ('kumo-api @ git+https://github.com/kumo-ai/'
                f'kumo-api.git@v{_KUMO_API_VER}')
else:
    kumo_api = f'kumo-api>={_KUMO_API_VER},<{_KUMO_API_MAJOR + 1}.0.0'

if bool(int(os.getenv('WITH_KUMOLIB', '1'))):
    kwargs = dict(cmake_source_dir='.')
else:
    kwargs = dict()

setup(
    install_requires=[
        "numpy",
        "pandas",
        "pyarrow>=8.0.0",
        "python-dateutil",
        "requests>=2.28.2",
        "urllib3",
        "typing_extensions>=4.5.0",
        kumo_api,
        "sdfm-connectors>=0.1,<1",
        "antlr4-python3-runtime==4.9.3",
        "rich>=9.0.0",
        "jinja2",
    ],
    **kwargs,
)

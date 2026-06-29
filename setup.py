import os

from setuptools import setup

kumo_api_ver = 'v0.92.0'
if int(os.getenv('KUMO_SDK_RELEASE', '0')):
    api_ver = kumo_api_ver[1:]
    major, minor, patch = api_ver.split('.')
    kumo_api = f'kumo-api>={api_ver},<{int(major)+1}.0.0'
elif kumo_api_path := os.getenv('KUMO_API_PATH'):
    kumo_api = f'kumo-api @ file://{os.path.abspath(kumo_api_path)}'
elif github_token := os.getenv('GITHUB_TOKEN'):
    kumo_api = f'kumo-api@git+https://{github_token}@github.com/kumo-ai/kumo-api.git@{kumo_api_ver}#egg=kumoapi'  # noqa: E501
else:
    kumo_api = f'kumo-api@git+ssh://git@github.com/kumo-ai/kumo-api.git@{kumo_api_ver}#egg=kumoapi'  # noqa: E501

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
        "rich>=9.0.0",
        "jinja2",
    ],
    **kwargs,
)

# Building PyPI Package

## Setup
1. Generate token on PyPI and copy to `.pypirc`

```
[pypi]
  username = __token__
  password = $TOKEN
```

2. Install build dependencies

```
pip install setuptools wheel twine
pip install --upgrade build
```
## Release

1. rev version number in project.toml, commit and push

2. build and upload

```
python -m build
twine upload dist/*
```

Build Your Package:

Run the following to generate distribution archives:

```bash
rm dist/*
python setup.py sdist bdist_wheel
```

Upload Your Package to TestPyPI (Optional):

Before uploading to PyPI, it's a good idea to test the upload on TestPyPI:

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Then, try installing your package from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pathologyfoundation
```


Alternatively, try to install from local and test the package:

```bash
cd ..
pip install ./pathologyfoundation/
```

Upload Your Package to PyPI:

If everything looks good, you can upload your package to PyPI:


```bash
twine upload dist/*
```

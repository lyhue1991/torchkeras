### Upload to  Pypi

```python
%run setup.py sdist bdist_wheel
```

```python
#!pip install --user --upgrade twine
```

```python
# upload the repository to pypi (test)
#!python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

```python
# install the repository from pypi (test)
#!python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps torchkeras
```

```python

```

### Upload to github

```python
!git config --global user.name "lyhue1991"
!git config --global user.email "lyhue1991@163.com"
```

```python
#!git reset  b37a9b8
```

```python
!git add -A
```

```python
!git commit -m"3.7.2"
```

```python
#!git config pull.rebase true
```

```python
!git pull origin master 
```

```python
!git remote remove origin 
```

```python
!git remote add origin git@github.com:lyhue1991/torchkeras.git
```

```python
!git push origin master 
```

```python
!git reflog 
```

```python

```

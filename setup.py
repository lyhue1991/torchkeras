# -*- coding:utf-8 -*-
import re
from pathlib import Path
from setuptools import find_packages, setup, find_namespace_packages

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / "README_en.md").read_text(encoding="utf-8")

def get_version():
    file = PARENT/'torchkeras/__init__.py'
    return re.search(r'__version__="(.*)"', file.read_text(encoding="utf-8"),re.M)[1]

setup(
    name="torchkeras",
    version=get_version(),
    author="lyhue1991,octopus,Laugh",
    author_email="lyhue1991@163.com",
    description="pytorch❤️keras",
    long_description=README,
    install_requires=[     
         #'accelerate>=0.20.3',
         #'tqdm',
         #'torch',
         #'torchmetrics',
         #'plotly',
         #'wandb'
       ],
    long_description_content_type="text/markdown",
    url="https://github.com/lyhue1991/torchkeras",
    packages=find_namespace_packages(exclude=['torchkeras.assets','data']),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="vlog, deep-learning, DL, pytorch, torch, keras",
    python_requires='>=3.5',
    # 确保没有使用 license_file 或 metadata_version 参数
)

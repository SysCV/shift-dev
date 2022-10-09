"""Package setup."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shift-dev",
    version="1.0.0",
    author="Tao Sun",
    author_email="taosun47@ethz.ch",
    description="SHIFT Dataset Devkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.vis.xyz/shift/",
    project_urls={
        "Documentation": "https://www.vis.xyz/shift/",
        "Source": "https://github.com/SysCV/shift-dev",
        "Tracker": "https://github.com/SysCV/shift-dev/issues",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "h5py",
        "matplotlib",
        "numpy",
        "pillow",
        "pycocotools",
        "scalabel @ git+https://github.com/scalabel/scalabel.git",
        "scikit-image",
        "pyyaml",
        "tqdm",
        "pydantic",
        "opencv-python",
    ],
    include_package_data=True,
)
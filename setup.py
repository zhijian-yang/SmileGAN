import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SmileGAN",
    version="0.0.4",
    author="zhijian.yang",
    author_email="zhijianyang@outlook.com",
    description="A python implementation of Smile-GAN for semisupervised clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nyuyzj/Smile-GAN",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)


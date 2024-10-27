from setuptools import setup, find_packages

setup(
    name="self-consistency",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "tqdm>=4.65.0",
    ],
    author="Yehor Duma",
    author_email="yegorduma@gmail.com",
    description="""Implementation of Wang, Xuezhi, et al. "Self-consistency improves chain of thought reasoning in language models. 
         "arXiv preprint arXiv:2203.11171 (2022).""",
    long_description=open("README.md").read(),
)
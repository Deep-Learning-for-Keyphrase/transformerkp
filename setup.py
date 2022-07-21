from setuptools import find_packages, setup


setup(
    name="transformerkp",
    version="0.0.1",
    author="Amardeep Kumar || Debanjan Mahata",
    author_email="kumaramardipsingh@gmail.com, debanjanmahata85@gmail.com",
    description="A transformer based deep learning library for keyphrase extraction and generation",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Deep-Learning-for-Keyphrase/transformerkp",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="MIT License",
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "requests",
        "tqdm>=4.47.0",
        "regex",
        "transformers>=4.6.0",
        "datasets",
        "scipy",
        "scikit-learn",
        "seqeval",
        "torch",
        "tensorflow>=2.0",
        "tensorboard",
        "pandas",
        "tokenizers",
    ]
)

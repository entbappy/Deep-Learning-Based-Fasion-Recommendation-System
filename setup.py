from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="Bappy Ahmed",
    description="A small package for Fasion Recommendation system Deep Learning Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/entbappy/",
    author_email="entbappy73@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        'tensorflow==2.3.1',
        'PyYAML',
        'scikit-learn',
        'tqdm',
        'streamlit',
        'opencv-python',
        'bing-image-downloader'
    ]
)
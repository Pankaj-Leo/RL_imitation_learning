"""
Setup script for imitation learning package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="imitation-learning",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-quality implementations of fundamental imitation learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/imitation-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "jax": [
            "jax>=0.4.13",
            "jaxlib>=0.4.13",
            "flax>=0.7.0",
            "optax>=0.1.7",
        ],
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
)

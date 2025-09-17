"""
Setup configuration for Layerwise-Adapter package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="layerwise-adapter",
    version="2.0.0",
    author="Layerwise-Adapter Team", 
    author_email="your-email@example.com",
    description="Knowledge Distillation for LLM Recommendation Systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/GeoffreyWang1117/Intelligent-Recommender",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0", 
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "experiments": [
            "jupyter>=1.0",
            "matplotlib>=3.5",
            "seaborn>=0.11",
            "tensorboard>=2.8",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "layerwise-distill=experiments.distillation_experiment:main",
            "layerwise-recommend=experiments.recommendation_benchmark:main",
        ],
    },
    
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/GeoffreyWang1117/Intelligent-Recommender/issues",
        "Source": "https://github.com/GeoffreyWang1117/Intelligent-Recommender",
        "Documentation": "https://github.com/GeoffreyWang1117/Intelligent-Recommender/blob/main/docs/",
    },
)

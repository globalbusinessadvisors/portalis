"""
Setup script for Portalis NeMo Integration

Install with:
    pip install -e .              # Basic installation
    pip install -e ".[cuda]"      # With CUDA support
    pip install -e ".[dev]"       # With development tools
    pip install -e ".[dev,cuda]"  # All dependencies
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="portalis-nemo-integration",
    version="0.1.0",
    author="Portalis Team",
    author_email="team@portalis.ai",
    description="NeMo-based Python to Rust translation engine for Portalis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/portalis/portalis",
    project_urls={
        "Bug Tracker": "https://github.com/portalis/portalis/issues",
        "Documentation": "https://portalis.readthedocs.io",
        "Source Code": "https://github.com/portalis/portalis",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Compilers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.10",
    install_requires=[
        "nemo-toolkit[all]>=1.22.0",
        "nvidia-pytriton>=0.4.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pydantic>=2.5.0",
        "libcst>=1.1.0",
        "astor>=0.8.1",
        "typing-extensions>=4.8.0",
        "loguru>=0.7.2",
        "pyyaml>=6.0.1",
        "jsonschema>=4.20.0",
        "tenacity>=8.2.3",
    ],
    extras_require={
        "cuda": [
            "cupy-cuda12x>=12.3.0",
            "nvidia-cuda-runtime-cu12>=12.3.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "hypothesis>=6.92.0",
            "black>=23.12.0",
            "ruff>=0.1.9",
            "mypy>=1.7.1",
            "ipython>=8.18.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "portalis-translate=src.translation.translator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
)

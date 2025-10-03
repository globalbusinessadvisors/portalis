"""
Setup configuration for Portalis DGX Cloud
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="portalis-dgx-cloud",
    version="1.0.0",
    description="NVIDIA DGX Cloud integration for Portalis translation pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Portalis Team",
    author_email="ml-ops@portalis.ai",
    url="https://github.com/your-org/portalis",
    license="MIT",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=">=3.10",

    install_requires=[
        "ray[default]>=2.9.0",
        "ray[tune]>=2.9.0",
        "boto3>=1.34.0",
        "redis>=5.0.1",
        "prometheus-client>=0.19.0",
        "psutil>=5.9.6",
        "pynvml>=11.5.0",
        "pydantic>=2.5.0",
        "loguru>=0.7.2",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "tenacity>=8.2.3",
        "requests>=2.31.0",
        "aiohttp>=3.9.1",
    ],

    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-mock>=3.12.0",
            "black>=23.12.0",
            "ruff>=0.1.9",
            "mypy>=1.7.1",
        ],
        "monitoring": [
            "grafana-api>=1.0.3",
            "prometheus-api-client>=0.5.3",
        ],
    },

    entry_points={
        "console_scripts": [
            "portalis-dgx=cli:main",
            "portalis-cost=cost.optimizer:main",
        ],
    },

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    keywords="nvidia dgx cloud distributed translation ray python rust",

    project_urls={
        "Documentation": "https://github.com/your-org/portalis/tree/main/dgx-cloud",
        "Source": "https://github.com/your-org/portalis",
        "Bug Reports": "https://github.com/your-org/portalis/issues",
    },
)

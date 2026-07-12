from setuptools import setup, find_packages

setup(
    name="kv_compression",
    version="0.1.0",
    description="Entropy-Driven KV Cache Compression",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.57.0",
        "torch>=2.11.0",
        "numpy>=1.26.0",
    ],
)

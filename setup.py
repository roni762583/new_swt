#!/usr/bin/env python3
"""
Setup script for SWT Phase 2 validation
"""

from setuptools import setup, find_packages

setup(
    name="swt-phase2",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.3",
        "torch>=2.0.1",
        "scipy>=1.10.1", 
        "pandas>=2.0.3",
        "pydantic>=2.3.0",
        "pyyaml>=6.0.1",
    ]
)
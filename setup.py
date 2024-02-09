from distutils.core import setup, Extension
import numpy as np

action_distributions_extension = Extension(
    "_mbag_action_distributions",
    sources=[
        "mbag/agents/_mbag_action_distributionsmodule.c",
    ],
    include_dirs=[np.get_include()],
)

setup(
    name="mbag",
    packages=[
        "mbag",
    ],
    ext_modules=[action_distributions_extension],
    version="0.0.1",
    author="Cassidy Laidlaw",
    author_email="cassidy_laidlaw@berkeley.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

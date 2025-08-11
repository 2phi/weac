"""
Snow grain types and hand hardness values for type annotations.

These values are used in Pydantic models for validation and correspond to the
parameterizations available in geldsetzer.py.
"""

from typing import Literal

# Grain types from SnowPilot notation (keys from GRAIN_TYPE in geldsetzer.py)
GRAIN_TYPES = Literal[
    "DF",
    "DFbk",
    "DFdc",
    "DH",
    "DHch",
    "DHcp",
    "DHla",
    "DHpr",
    "DHxr",
    "FC",
    "FCsf",
    "FCso",
    "FCxr",
    "IF",
    "IFbi",
    "IFic",
    "IFil",
    "IFrc",
    "IFsc",
    "MF",
    "MFcl",
    "MFcr",
    "MFpc",
    "MFsl",
    "PP",
    "PPco",
    "PPgp",
    "PPhl",
    "PPip",
    "PPir",
    "PPnd",
    "PPpl",
    "PPrm",
    "PPsd",
    "RG",
    "RGlr",
    "RGsr",
    "RGwp",
    "RGxf",
    "SH",
    "SHcv",
    "SHsu",
    "SHxr",
]

# Hand hardness values from field notation (keys from HAND_HARDNESS in geldsetzer.py)
HAND_HARDNESS_VALUES = Literal[
    "F-",
    "F",
    "F+",
    "4F-",
    "4F",
    "4F+",
    "1F-",
    "1F",
    "1F+",
    "P-",
    "P",
    "P+",
    "K-",
    "K",
    "K+",
    "I-",
    "I",
    "I+",
]

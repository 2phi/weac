"""
Hand hardness + Grain Type Parameterization to Density
according to Geldsetzer & Jamieson (2000)
`https://arc.lib.montana.edu/snow-science/objects/issw-2000-121-127.pdf`

Inputs:
Hand Hardness + Grain Type
Output:
Density [kg/m^3]
"""

from typing import Tuple

DENSITY_PARAMETERS = {
    "!skip": (0, 0),
    "PP": (45, 36),
    "PPgp": (83, 37),
    "DF": (65, 36),
    "FCmx": (56, 64),
    "FC": (112, 46),
    "DH": (185, 25),
    "RGmx": (91, 42),
    "RG": (154, 1.51),
    "MFCr": (292.25, 0),
}

# Map SnowPilot grain type to those we know
GRAIN_TYPE = {
    "": "!skip",
    "DF": "DF",
    "DFbk": "DF",
    "DFdc": "DF",
    "DH": "DH",
    "DHch": "DH",
    "DHcp": "DH",
    "DHla": "DH",
    "DHpr": "DH",
    "DHxr": "DH",
    "FC": "FC",
    "FCsf": "FCmx",
    "FCso": "FCmx",
    "FCxr": "FCmx",
    "IF": "MFCr",
    "IFbi": "MFCr",
    "IFic": "MFCr",
    "IFil": "MFCr",
    "IFrc": "MFCr",
    "IFsc": "MFCr",
    "MF": "MFCr",
    "MFcl": "MFCr",
    "MFcr": "MFCr",
    "MFpc": "MFCr",
    "MFsl": "MFCr",
    "PP": "PP",
    "PPco": "PP",
    "PPgp": "PPgp",
    "PPhl": "PP",
    "PPip": "PP",
    "PPir": "PP",
    "PPnd": "PP",
    "PPpl": "PP",
    "PPrm": "PP",
    "PPsd": "PP",
    "RG": "RG",
    "RGlr": "RGmx",
    "RGsr": "RGmx",
    "RGwp": "RGmx",
    "RGxf": "RGmx",
    "SH": "!skip",
    "SHcv": "!skip",
    "SHsu": "!skip",
    "SHxr": "!skip",
}

# Translate hand hardness to numerical values
HAND_HARDNESS = {
    "": "!skip",
    "F-": 0.67,
    "F": 1,
    "F+": 1.33,
    "4F-": 1.67,
    "4F": 2,
    "4F+": 2.33,
    "1F-": 2.67,
    "1F": 3,
    "1F+": 3.33,
    "P-": 3.67,
    "P": 4,
    "P+": 4.33,
    "K-": 4.67,
    "K": 5,
    "K+": 5.33,
    "I-": 5.67,
    "I": 6,
    "I+": 6.33,
}


def compute_density(grainform: str, hardness: str | Tuple[str, str]) -> float:
    """
    Geldsetzer & Jamieson (2000)
    `https://arc.lib.montana.edu/snow-science/objects/issw-2000-121-127.pdf`
    """
    # Adaptation based on CAAML profiles (which sometimes provide top and bottom hardness)
    print(grainform, hardness)
    if isinstance(hardness, tuple):
        hardness_top, hardness_bottom = hardness
        hardness_value = (
            HAND_HARDNESS[hardness_top] + HAND_HARDNESS[hardness_bottom]
        ) / 2
    else:
        hardness_value = HAND_HARDNESS[hardness]
    grain_type = GRAIN_TYPE[grainform]
    a, b = DENSITY_PARAMETERS[grain_type]

    if grain_type == "RG":
        # Special computation for 'RG' grain form
        return a + b * (hardness_value**3.15)
    else:
        return a + b * hardness_value

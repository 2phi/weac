"""
Hand hardness + Grain Type Parameterization to Density
according to Geldsetzer & Jamieson (2000)
`https://arc.lib.montana.edu/snow-science/objects/issw-2000-121-127.pdf`

Inputs:
Hand Hardness + Grain Type
Output:
Density [kg/m^3]
"""

SKIP_VALUE = "!skip"


DENSITY_PARAMETERS = {
    SKIP_VALUE: (0, 0),
    "SH": (125, 0),  # 125 kg/m^3 so that bergfeld is E~1.0
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
    "": SKIP_VALUE,
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
    "gp": "PPgp",
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
    "SH": "SH",
    "SHcv": "SH",
    "SHsu": "SH",
    "SHxr": "SH",
    "WG": "WG",
}

# Translate hand hardness to numerical values
HAND_HARDNESS = {
    "": SKIP_VALUE,
    "F-": 0.67,
    "F": 1,
    "F+": 1.33,
    "F-4F": 1.5,
    "4F-": 1.67,
    "4F": 2,
    "4F+": 2.33,
    "4F-1F": 2.5,
    "1F-": 2.67,
    "1F": 3,
    "1F+": 3.33,
    "1F-P": 3.5,
    "P-": 3.67,
    "P": 4,
    "P+": 4.33,
    "P-K": 4.5,
    "K-": 4.67,
    "K": 5,
    "K+": 5.33,
    "K-I": 5.5,
    "I-": 5.67,
    "I": 6,
    "I+": 6.33,
}

GRAIN_TYPE_TO_DENSITY = {
    "PP": 84.9,
    "PPgp": 162.3,
    "DF": 136.3,
    "RG": 247.4,
    "RGmx": 220.6,
    "FC": 248.2,
    "FCmx": 288.8,
    "DH": 252.8,
    "WG": 254.3,
    "MFCr": 292.3,
    "SH": 125,
}

HAND_HARDNESS_TO_DENSITY = {
    "F-": 71.7,
    "F": 103.7,
    "F+": 118.4,
    "F-4F": 123.15,
    "4F-": 127.9,
    "4F": 158.2,
    "4F+": 163.7,
    "4F-1F": 176.15,
    "1F-": 188.6,
    "1F": 208,
    "1F+": 224.4,
    "1F-P": 238.6,
    "P-": 252.8,
    "P": 275.9,
    "P+": 314.6,
    "P-K": 336.85,
    "K-": 359.1,
    "K": 347.4,
    "K+": 407.8,
    "K-I": 407.8,
    "I-": 407.8,
    "I": 407.8,
    "I+": 407.8,
}


def compute_density(grainform: str | None, hardness: str | None) -> float:
    """
    Geldsetzer & Jamieson (2000)
    `https://arc.lib.montana.edu/snow-science/objects/issw-2000-121-127.pdf`
    """
    # Adaptation based on CAAML profiles (which sometimes provide top and bottom hardness)
    if hardness is None and grainform is None:
        raise ValueError("Provide at least one of grainform or hardness")
    if hardness is None:
        grain_type = GRAIN_TYPE[grainform]
        return GRAIN_TYPE_TO_DENSITY[grain_type]
    if grainform is None:
        return HAND_HARDNESS_TO_DENSITY[hardness]

    hardness_value = HAND_HARDNESS[hardness]
    grain_type = GRAIN_TYPE[grainform]
    a, b = DENSITY_PARAMETERS[grain_type]

    if grain_type == SKIP_VALUE:
        raise ValueError(f"Grain type is {SKIP_VALUE}")
    if hardness_value == SKIP_VALUE:
        raise ValueError(f"Hardness value is {SKIP_VALUE}")

    if grain_type == "RG":
        # Special computation for 'RG' grain form
        rho = a + b * (hardness_value**3.15)
    else:
        rho = a + b * hardness_value
    return rho

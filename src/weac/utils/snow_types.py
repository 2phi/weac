"""
Snow grain types and hand hardness values.

These values are used in Pydantic models for validation and correspond to the
parameterizations available in `geldsetzer.py`.
"""

from enum import Enum


class GrainType(str, Enum):
    """SnowPilot grain type codes (see `geldsetzer.GRAIN_TYPE`)."""

    DF = "DF"
    DFbk = "DFbk"
    DFdc = "DFdc"
    DH = "DH"
    DHch = "DHch"
    DHcp = "DHcp"
    DHla = "DHla"
    DHpr = "DHpr"
    DHxr = "DHxr"
    FC = "FC"
    FCsf = "FCsf"
    FCso = "FCso"
    FCxr = "FCxr"
    IF = "IF"
    IFbi = "IFbi"
    IFic = "IFic"
    IFil = "IFil"
    IFrc = "IFrc"
    IFsc = "IFsc"
    MF = "MF"
    MFcl = "MFcl"
    MFcr = "MFcr"
    MFpc = "MFpc"
    MFsl = "MFsl"
    PP = "PP"
    PPco = "PPco"
    PPgp = "PPgp"
    PPhl = "PPhl"
    PPip = "PPip"
    PPir = "PPir"
    PPnd = "PPnd"
    PPpl = "PPpl"
    PPrm = "PPrm"
    PPsd = "PPsd"
    RG = "RG"
    RGlr = "RGlr"
    RGsr = "RGsr"
    RGwp = "RGwp"
    RGxf = "RGxf"
    SH = "SH"
    SHcv = "SHcv"
    SHsu = "SHsu"
    SHxr = "SHxr"


class HandHardness(str, Enum):
    """Field hand hardness codes (see `geldsetzer.HAND_HARDNESS`).

    Enum member names avoid starting with digits and special characters.
    """

    Fm = "F-"
    F = "F"
    Fp = "F+"
    F_4F = "F-4F"
    _4Fm = "4F-"
    _4F = "4F"
    _4Fp = "4F+"
    _4F_1F = "4F-1F"
    _1Fm = "1F-"
    _1F = "1F"
    _1Fp = "1F+"
    _1F_P = "1F-P"
    Pm = "P-"
    P = "P"
    Pp = "P+"
    P_K = "P-K"
    Km = "K-"
    K = "K"
    Kp = "K+"
    K_I = "K-I"
    Im = "I-"
    I = "I"
    Ip = "I+"

from enum import Enum, auto


class SpectrumType(Enum):
    VCD = auto()
    IR = auto()
    ECD = auto()
    UV = auto()


def string_to_spectrum_type(string: str) -> SpectrumType:
    if string == 'VCD':
        return SpectrumType.VCD
    if string == 'IR':
        return SpectrumType.IR
    if string == 'ECD':
        return SpectrumType.ECD
    if string == 'UV':
        return SpectrumType.UV
    raise ValueError(f'Unknown spectrum type: {string}')

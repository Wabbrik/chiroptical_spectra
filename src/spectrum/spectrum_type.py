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


title_by_type = {
    SpectrumType.VCD: "$\Delta \epsilon$(M$^{-1} \cdot$ cm$^{-1}$)",
    SpectrumType.IR: "$\epsilon$(M$^{-1} \cdot$ cm$^{-1}$)",
    SpectrumType.ECD: "$\Delta \epsilon$(M$^{-1} \cdot$ cm$^{-1}$)",
    SpectrumType.UV: "$\epsilon$(M$^{-1} \cdot$ cm$^{-1}$)",
}

prefix_by_type = {
    SpectrumType.VCD: "rs_",
    SpectrumType.IR: "ds_",
    SpectrumType.ECD: "ecd_",
    SpectrumType.UV: "uv_",
}

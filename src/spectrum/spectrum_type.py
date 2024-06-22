from enum import Enum, auto

from broadening.broadening import ecd_broaden, ir_broaden, uv_broaden, vcd_broaden


class SpectrumType(Enum):
    ROA = auto()
    VCD = auto()
    IR = auto()
    ECD = auto()
    UV = auto()


def string_to_spectrum_type(string: str) -> SpectrumType:
    if string == "ROA":
        return SpectrumType.ROA
    if string == "VCD":
        return SpectrumType.VCD
    if string == "IR":
        return SpectrumType.IR
    if string == "ECD":
        return SpectrumType.ECD
    if string == "UV":
        return SpectrumType.UV
    raise ValueError(f"Unknown spectrum type: {string}")


title_by_type = {
    SpectrumType.ROA: "$\Delta \epsilon$(M$^{-1} \cdot$ cm$^{-1}$)",
    SpectrumType.VCD: "$\Delta \epsilon$(M$^{-1} \cdot$ cm$^{-1}$)",
    SpectrumType.IR: "$\epsilon$(M$^{-1} \cdot$ cm$^{-1}$)",
    SpectrumType.ECD: "$\Delta \epsilon$(M$^{-1} \cdot$ cm$^{-1}$)",
    SpectrumType.UV: "$\epsilon$(M$^{-1} \cdot$ cm$^{-1}$)",
}

prefix_by_type = {
    SpectrumType.ROA: "roa_",
    SpectrumType.VCD: "rs_",
    SpectrumType.IR: "ds_",
    SpectrumType.ECD: "ecd_",
    SpectrumType.UV: "uv_",
}

broaden_funcs = {
    SpectrumType.ROA: ir_broaden,
    SpectrumType.VCD: vcd_broaden,
    SpectrumType.IR: ir_broaden,
    SpectrumType.ECD: ecd_broaden,
    SpectrumType.UV: uv_broaden,
}

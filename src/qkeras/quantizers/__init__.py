from .quantized_bits import quantized_bits

__all__ = ['quantized_bits']


def get_quantizer(str_conf: str):
    name = str_conf.split('(', 1)[0]
    if name in globals():
        return globals()[name].from_string(str_conf)
    raise ValueError(f"Unknown quantizer: {name}")

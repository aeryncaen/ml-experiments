from .s6 import S6

try:
    from .triton_s6 import TritonS6, HAS_TRITON
except ImportError:
    TritonS6 = None
    HAS_TRITON = False

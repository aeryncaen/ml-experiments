"""
S6: Unified Sequence Block (USB) Implementation

A fused architecture combining SSM-style scans, attention, and MLP
into a single expand-process-contract block.
"""

from .usb_block import USBBlock, USBConfig

__all__ = ["USBBlock", "USBConfig"]

# Frame format conversion
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

"""Shared, framework-agnostic pixel-format conversion for task code.

Tasks that draw overlays work in RGB and convert to the target video pixel
format before handing the frame back to the element shell for writing.
"""


def to_format(rgb, fmt):
    """Convert an RGB numpy array to the target video pixel format `fmt`."""
    import cv2
    import numpy as np

    if fmt == "RGB":
        return rgb
    elif fmt == "BGR":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif fmt == "RGBA":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    elif fmt == "BGRA":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
    elif fmt == "ARGB":
        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        return np.roll(rgba, 1, axis=-1)
    elif fmt == "ABGR":
        bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
        return np.roll(bgra, 1, axis=-1)
    else:
        return rgb

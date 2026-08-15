"""PNG output.

Converts binary mask of watershed extraction to RGBA PNG for web map overlay.
Basin cells are semi-transparent blue, outside basin is fully transparent.
Intended to be used with bbox in Leaflet ImageOverlay / MapLibre ImageSource.
"""

import numpy as np
from PIL import Image

from watershed.models import BasinResult

# Basin color: rgba(0, 100, 255, 128) = semi-transparent blue
_BASIN_RGBA = (0, 100, 255, 128)


def write_png(result: BasinResult, output_path: str) -> None:
    """Write watershed mask to RGBA PNG file.

    Args:
        result: Watershed extraction result (uses mask)
        output_path: Output PNG path (e.g., /tmp/basin.png)
    """
    mask = result.mask  # uint8: 1=inside basin, 0=outside basin
    h, w = mask.shape

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask == 1] = _BASIN_RGBA

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(output_path, format="PNG")

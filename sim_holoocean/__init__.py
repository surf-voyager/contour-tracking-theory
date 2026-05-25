"""sim_holoocean — Layer 2 (HoloOcean photorealistic) contour-tracking simulator.

See sim_holoocean/README.md for what this layer is and how to run it.

This package is self-contained: it does NOT import from the Layer-1 sim_python
package (and vice versa). The ported controllers re-implement the Layer-1 control
law byte-for-byte so the two engines compare the same controller.
"""

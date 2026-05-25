"""Monte-Carlo framework for sim_python.

Latin-Hypercube sampler + parallel dispatcher + analysis helpers.

Public surface
--------------
- sampler.latin_hypercube_sample / sampler.adaptive_boundary_refine
- dispatcher.run_batch (CLI entry point)
- analysis.aggregator.load_batch / analysis.aggregator.compute_per_cell
- analysis.phase_diagram.plot_2d_slice
- analysis.check_phenomena (6 stubs returning NotImplementedError)
"""

__all__ = ["sampler", "dispatcher", "analysis"]

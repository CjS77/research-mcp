"""research_kb — a fidelity-first hybrid-retrieval knowledge base served over MCP.

The package implements a fidelity-first retrieval pipeline:
deterministic + LLM extraction with a divergence cross-check, provenance-carrying
chunking, hybrid (semantic + BM25 + RRF) search, a citation graph, an eval harness,
and an MCP server / CLI as the interface surface.
"""

__version__ = "0.1.0"

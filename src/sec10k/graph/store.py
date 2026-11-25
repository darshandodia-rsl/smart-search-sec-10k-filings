"""The connection to the Neo4j graph database.

What it does: opens Neo4j through LlamaIndex's `Neo4jPropertyGraphStore`, using the
settings from `config.py`.

Why it is needed: every part of the project that touches the graph (writing the structure,
embedding chunks, retrieval, the API) opens it through this one function, so they all use
the same connection settings.
"""

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from sec10k.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def get_graph_store(refresh_schema: bool = False) -> Neo4jPropertyGraphStore:
    """Connect to Neo4j and return the graph store.

    On first use it creates a uniqueness rule on `__Node__.id`. That rule is what lets us
    write nodes with MERGE-by-id: re-running a writer updates nodes instead of duplicating
    them, and the lookups stay fast.

    `refresh_schema` controls whether LlamaIndex reads the whole graph's structure when it
    connects. Writers don't need that, and it gets slow on a big graph, so it is off by
    default. Text-to-Cypher (Phase 6) needs the schema and should turn it on.
    """
    return Neo4jPropertyGraphStore(
        username=NEO4J_USER, password=NEO4J_PASSWORD, url=NEO4J_URI,
        refresh_schema=refresh_schema,
    )

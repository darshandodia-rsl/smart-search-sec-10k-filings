"""Connection to Neo4j through LlamaIndex's Neo4jPropertyGraphStore.

One place to open the store, so ingestion, retrieval and the API all use the same settings.
"""

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from sec10k.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def get_graph_store(refresh_schema: bool = False) -> Neo4jPropertyGraphStore:
    """Open the store. On first use it creates a uniqueness constraint on __Node__.id
    (what makes our MERGE-by-id writes idempotent and fast).
    refresh_schema=False: skip reading the whole graph's schema on connect; writers don't
    need it, and it gets slow on a large graph (text-to-Cypher in Phase 6 will want it)."""
    return Neo4jPropertyGraphStore(
        username=NEO4J_USER, password=NEO4J_PASSWORD, url=NEO4J_URI,
        refresh_schema=refresh_schema,
    )

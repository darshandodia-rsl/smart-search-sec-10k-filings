"""Extraction tiers: find entities and relationships in chunk text.

Holds the cheap, no-LLM tiers (dictionary and pattern matching, local NLP) and the
schema-guided LLM extraction (`llm_extract.py`). The cheap tiers run on every chunk and also decide which
chunks are worth sending to the LLM.
"""

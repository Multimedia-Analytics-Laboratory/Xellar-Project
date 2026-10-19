#!/usr/bin/env python3
"""Build an official KAG 0.8 memory graph from masked KGML JSON exports.

The stock KAG external graph loader contacts the OpenSPG schema service during
initialization.  This builder deliberately uses KAG's official graph model,
vectorization interface, MemoryGraphWriter standardization, and MemoryGraph
storage directly, allowing the same solver stack to run without a daemon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(vector):
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not norm:
        return list(vector)
    return [float(value) / norm for value in vector]


def _batched(items, size):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--graph-path", default="./ckpt/MemoryGraph")
    parser.add_argument("--manifest")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing-embeddings", action="store_true")
    args = parser.parse_args()

    nodes_path = Path(args.nodes).resolve()
    edges_path = Path(args.edges).resolve()
    graph_path = Path(args.graph_path).resolve()
    graph_file = graph_path / "graph"
    reusable_vectors = {}
    if args.reuse_existing_embeddings and graph_file.exists():
        import igraph as ig

        existing_graph = ig.Graph.Read(str(graph_file), "picklez")
        for vertex in existing_graph.vs:
            attributes = vertex.attributes()
            vector = attributes.get("_name_vector")
            if vector:
                reusable_vectors[attributes["id"]] = vector
    if graph_file.exists() and not args.overwrite:
        raise FileExistsError(f"memory graph already exists: {graph_file}")
    if args.overwrite and graph_path.exists():
        shutil.rmtree(graph_path)
    graph_path.mkdir(parents=True, exist_ok=True)

    # Import only after entering a directory containing kag_config.yaml. KAG
    # resolves !ENV values at import time; the key is never written to disk.
    import kag
    from kag.builder.component.writer.memory_graph_writer import MemoryGraphWriter
    from kag.builder.model.sub_graph import Edge, Node, SubGraph
    from kag.common.conf import KAG_CONFIG
    from kag.common.graphstore.memory_graph import MemoryGraph
    from kag.interface import VectorizeModelABC

    if not os.getenv("VECTORIZE_MODEL_API_KEY"):
        raise RuntimeError("VECTORIZE_MODEL_API_KEY is required")

    node_rows = json.loads(nodes_path.read_text(encoding="utf-8"))
    edge_rows = json.loads(edges_path.read_text(encoding="utf-8"))
    nodes = [Node.from_dict(row) for row in node_rows]
    edges = [Edge.from_dict(row) for row in edge_rows]

    vectorizer = VectorizeModelABC.from_config(
        KAG_CONFIG.all_config["vectorize_model"]
    )
    started_at = time.time()
    embedding_dimensions = None
    missing_nodes = []
    for node in nodes:
        vector = reusable_vectors.get(node.id)
        if vector:
            node.properties["_name_vector"] = list(vector)
            embedding_dimensions = embedding_dimensions or len(vector)
        else:
            missing_nodes.append(node)
    for start, batch in _batched(missing_nodes, args.batch_size):
        vectors = vectorizer.vectorize([node.name for node in batch])
        if vectors is None or len(vectors) != len(batch):
            raise RuntimeError(f"embedding batch failed at node offset {start}")
        for node, vector in zip(batch, vectors):
            if not vector:
                raise RuntimeError(f"empty embedding for node {node.name!r}")
            normalized = _normalize(vector)
            if embedding_dimensions is None:
                embedding_dimensions = len(normalized)
            elif len(normalized) != embedding_dimensions:
                raise RuntimeError("inconsistent embedding dimensions")
            node.properties["_name_vector"] = normalized

    graph = SubGraph(nodes=nodes, edges=edges)
    graph = MemoryGraphWriter().standarlize_graph(graph)
    memory_graph = MemoryGraph(
        KAG_CONFIG.global_config.namespace, str(graph_path), vectorizer
    )
    # KAG's one-hop API distinguishes incoming and outgoing relations. Its
    # default empty memory graph is undirected, so initialize the backend as a
    # directed igraph before using the official upsert/dump implementation.
    import igraph as ig

    memory_graph._backend_graph = ig.Graph(directed=True)
    memory_graph.upsert_subgraph(graph.to_dict())
    memory_graph.dump()

    # Re-open the serialized graph in a fresh singleton key and validate basic
    # counts using python-igraph, without displaying configuration or secrets.
    stored = ig.Graph.Read(str(graph_file), "picklez")
    if stored.vcount() != len(nodes) or stored.ecount() != len(edges):
        raise RuntimeError(
            "serialized graph count mismatch: "
            f"{stored.vcount()}/{stored.ecount()} != {len(nodes)}/{len(edges)}"
        )

    manifest = {
        "upstream": "OpenSPG/KAG",
        "kag_version": getattr(kag, "__version__", None),
        "backend": "official_memory_graph",
        "namespace": KAG_CONFIG.global_config.namespace,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "embedding_model": KAG_CONFIG.all_config["vectorize_model"]["model"],
        "embedding_dimensions": embedding_dimensions,
        "embedding_batch_size": args.batch_size,
        "reused_embedding_count": len(nodes) - len(missing_nodes),
        "directed": stored.is_directed(),
        "elapsed_seconds": time.time() - started_at,
        "nodes_sha256": _sha256(nodes_path),
        "edges_sha256": _sha256(edges_path),
        "graph_path": str(graph_path),
        "graph_sha256": _sha256(graph_file),
    }
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else graph_path / "build_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "node_count": len(nodes),
                "edge_count": len(edges),
                "embedding_dimensions": embedding_dimensions,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

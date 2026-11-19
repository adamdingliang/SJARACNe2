"""Measure and summarize a directed consensus network from its output table."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


_IDENTIFIER_WHITESPACE = " \t\n\r\v\f"


@dataclass(frozen=True)
class DriverQC:
    driver: str
    target_count: int
    betweenness: float | None
    gene_symbol: str | None = None
    incoming_count: int = 0

    @property
    def total_count(self) -> int:
        return self.incoming_count + self.target_count


@dataclass(frozen=True)
class DegreeFitPoint:
    degree: int
    frequency: float
    fit_frequency: float
    fitted_frequency: float | None


@dataclass(frozen=True)
class NetworkQC:
    summary: dict[str, int | float | str | None]
    drivers: tuple[DriverQC, ...]
    degree_fit: tuple[DegreeFitPoint, ...]


def _requested_hubs(path: Path) -> tuple[str, ...]:
    # Match the inference parser: trim ASCII boundary whitespace and preserve
    # the first occurrence. Text mode accepts LF, CRLF and lone CR.
    with path.open(encoding="utf-8-sig") as handle:
        hubs = tuple(dict.fromkeys(line.strip(_IDENTIFIER_WHITESPACE) for line in handle))
    hubs = tuple(identifier for identifier in hubs if identifier)
    if not hubs:
        raise ValueError(f"Requested hub list has no identifiers: {path}")
    if any("\x00" in identifier for identifier in hubs):
        raise ValueError(f"Requested hub list contains a NUL byte: {path}")
    return hubs


def _read_expression_annotations(
    path: Path,
    requested: tuple[str, ...],
    *,
    network_ids: Iterable[str] | None,
) -> tuple[dict[str, str | None], int | None]:
    remaining = set(requested)
    symbols: dict[str, str | None] = {}
    expression_ids: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        header = handle.readline().rstrip("\r\n").split("\t", 2)
        if header[:2] != ["isoformId", "geneSymbol"]:
            raise ValueError(f"Expression table must start with isoformId and geneSymbol columns: {path}")
        for line in handle:
            columns = line.rstrip("\r\n").split("\t", 2)
            if len(columns) < 2:
                continue
            identifier = columns[0].strip(_IDENTIFIER_WHITESPACE)
            if network_ids is not None and identifier and "\x00" not in identifier:
                expression_ids.add(identifier)
            if identifier not in remaining:
                continue
            # As in consensus, use the first expression row for each ID.
            symbols[identifier] = columns[1].strip(_IDENTIFIER_WHITESPACE) or None
            remaining.remove(identifier)
            if not remaining and network_ids is None:
                break
    if network_ids is None:
        return symbols, None

    for identifier in network_ids:
        if identifier not in expression_ids:
            raise ValueError(f"Network ID {identifier!r} is absent from the expression TSV: {path}")
    return symbols, len(expression_ids)


def _read_graph(path: Path) -> tuple[csr_matrix, dict[str, int]]:
    vertices: dict[str, int] = {}
    # Packed index arrays keep parsing memory proportional to the edge count
    # without creating two Python integers and a tuple per edge.
    edges = array("q")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(rows, [])
        required = ("source", "target", "MI")
        if any(header.count(name) != 1 for name in required):
            raise ValueError(f"Network table must have one column each for source, target and MI: {path}")
        source_col, target_col, mi_col = (header.index(name) for name in required)
        for columns in rows:
            if not columns:
                continue
            location = f"{path}:{rows.line_num}"
            if len(columns) != len(header):
                raise ValueError(f"Network row has a different column count from its header: {location}")
            source = columns[source_col].strip(_IDENTIFIER_WHITESPACE)
            target = columns[target_col].strip(_IDENTIFIER_WHITESPACE)
            if not source or not target or "\x00" in source or "\x00" in target:
                raise ValueError(f"Network row has a missing or invalid source or target ID: {location}")
            if source == target:
                raise ValueError(f"Network contains a self-loop for {source!r}: {location}")
            try:
                mi = float(columns[mi_col])
            except ValueError as error:
                raise ValueError(f"Network row has an invalid MI value: {location}") from error
            if not math.isfinite(mi) or mi < 0:
                raise ValueError(f"Network MI must be finite and nonnegative: {location}")
            edges.append(vertices.setdefault(source, len(vertices)))
            edges.append(vertices.setdefault(target, len(vertices)))
    coordinates = np.frombuffer(edges, dtype=np.int64).reshape(-1, 2)
    edge_count = coordinates.shape[0]
    graph = csr_matrix(
        (np.ones(edge_count, dtype=np.uint8), (coordinates[:, 0], coordinates[:, 1])),
        shape=(len(vertices), len(vertices)),
    )
    if graph.nnz != edge_count:
        raise ValueError(f"Network contains duplicate directed edge pairs: {path}")
    return graph, vertices


def _degree_distribution(
    total_degree: np.ndarray,
) -> tuple[tuple[DegreeFitPoint, ...], float | None, float | None]:
    if total_degree.size == 0:
        return (), None, None
    frequencies = np.bincount(total_degree)[1:] / total_degree.size
    degrees = np.arange(1, frequencies.size + 1)
    # Match NetBID2 on the observed graph by adding one count to every
    # positive-degree bin, including gaps, before fitting in log space.
    fit_frequencies = frequencies + 1 / total_degree.size
    adjusted_r2 = slope = None
    fitted_frequencies: Sequence[float | None] = [None] * degrees.size
    if degrees.size >= 3 and np.ptp(frequencies) > 0:
        x = np.log10(degrees)
        y = np.log10(fit_frequencies)
        y_centered = y - y.mean()
        total_variation = float(np.dot(y_centered, y_centered))
        x_centered = x - x.mean()
        slope = float(np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered))
        predicted = y.mean() + slope * x_centered
        residual = y - predicted
        adjusted_r2 = float(1 - np.dot(residual, residual) / total_variation
                            * (degrees.size - 1) / (degrees.size - 2))
        fitted_frequencies = np.power(10.0, predicted).tolist()
    points = tuple(
        DegreeFitPoint(int(degree), float(frequency), float(fit_frequency), fitted)
        for degree, frequency, fit_frequency, fitted in zip(
            degrees, frequencies, fit_frequencies, fitted_frequencies, strict=True
        )
    )
    return points, adjusted_r2, slope


def _brandes(indptr, successors, sources):
    """Accumulate directed unweighted vertex betweenness for given sources."""
    size = len(indptr) - 1
    scores = np.zeros(size, dtype=np.float64)
    distance = np.empty(size, dtype=np.int64)
    paths = np.empty(size, dtype=np.float64)
    dependency = np.empty(size, dtype=np.float64)
    visited = np.empty(size, dtype=np.int64)
    for source in sources:
        distance.fill(-1)
        paths.fill(0)
        visited[0] = source
        distance[source] = 0
        paths[source] = 1
        head = 0
        tail = 1
        while head < tail:
            vertex = visited[head]
            head += 1
            next_distance = distance[vertex] + 1
            for edge in range(indptr[vertex], indptr[vertex + 1]):
                target = successors[edge]
                if distance[target] < 0:
                    distance[target] = next_distance
                    visited[tail] = target
                    tail += 1
                if distance[target] == next_distance:
                    paths[target] += paths[vertex]
        # The queue is in distance order. Walk it backwards, following only
        # edges in the shortest-path DAG. This needs no predecessor lists.
        # The source is first in the queue and cannot be an intermediate.
        for position in range(tail - 1, 0, -1):
            vertex = visited[position]
            if not math.isfinite(paths[vertex]):
                raise ValueError("Network shortest-path counts exceeded numeric precision")
            next_distance = distance[vertex] + 1
            value = 0.0
            for edge in range(indptr[vertex], indptr[vertex + 1]):
                target = successors[edge]
                if distance[target] == next_distance:
                    value += (paths[vertex] / paths[target]) * (1.0 + dependency[target])
            dependency[vertex] = value
            scores[vertex] += value
    return scores


@lru_cache(maxsize=1)
def _brandes_dispatcher():
    # Import and compile on the first calculation. Reuse the dispatcher
    # within this process without a disk cache.
    try:
        from numba import njit
    except ImportError as error:
        raise RuntimeError("Network QC requires Numba. Reinstall SJARACNe2 with its dependencies.") from error
    return njit(cache=False)(_brandes)


def _betweenness(
    graph: csr_matrix,
    vertices: dict[str, int],
    out_degree: np.ndarray,
) -> tuple[np.ndarray | None, str, int]:
    # Accumulate from every node with outgoing edges in stable ID order.
    # Sinks can end a path but contribute zero as starting nodes.
    sources = [idx for _, idx in sorted(vertices.items()) if out_degree[idx] > 0]
    node_count = graph.shape[0]
    if node_count < 3:
        return None, "unavailable", len(sources)
    scores = _brandes_dispatcher()(graph.indptr, graph.indices, np.asarray(sources, dtype=np.int64))
    scores /= (node_count - 1) * (node_count - 2)
    if not np.isfinite(scores).all():
        raise ValueError("Network shortest-path counts exceeded numeric precision")
    return scores, "exact", len(sources)


def compute_network_qc(
    network_file: str | Path,
    requested_hubs_file: str | Path,
    *,
    expression_gene_count: int | None = None,
    expression_file: str | Path | None = None,
) -> NetworkQC:
    """Summarize the final edge table and requested hubs for a QC report.

    Args:
        network_file: Consensus TSV with source, target and MI columns.
        requested_hubs_file: Input list with one requested driver per line.
        expression_gene_count: Number of unique validated expression IDs, when available.
        expression_file: Optional expression TSV with isoformId and geneSymbol
            columns. Without a validated count, enumerate unique IDs and verify
            that the expression table contains every network node.

    Returns:
        Network, driver and degree-distribution measurements. Undefined
        statistics are represented by None.

    Raises:
        OSError: An input file cannot be read.
        ValueError: The graph, expression header or gene count is invalid, or
            an edge endpoint is absent from the supplied expression table.
        RuntimeError: Numba is unavailable for betweenness calculation.
    """
    requested = _requested_hubs(Path(requested_hubs_file))
    graph, vertices = _read_graph(Path(network_file))
    node_count = graph.shape[0]
    if expression_gene_count is not None and (
        type(expression_gene_count) is not int or expression_gene_count < node_count
    ):
        raise ValueError("Expression gene count must be an integer at least as large as the number of network nodes")
    symbols = {}
    if expression_file is not None:
        symbols, scanned_gene_count = _read_expression_annotations(
            Path(expression_file), requested,
            network_ids=vertices if expression_gene_count is None else None,
        )
        if scanned_gene_count is not None:
            expression_gene_count = scanned_gene_count
    isolated_nodes = None if expression_gene_count is None else expression_gene_count - node_count
    out_degree = np.diff(graph.indptr)
    in_degree = np.bincount(graph.indices, minlength=node_count)
    driver_ids = [vertices.get(identifier) for identifier in requested]
    targets = [int(out_degree[idx]) if idx is not None else 0 for idx in driver_ids]
    active_drivers = sum(count > 0 for count in targets)
    degree_fit, adjusted_r2, slope = _degree_distribution(in_degree + out_degree)
    scores, method, eligible_sources = _betweenness(graph, vertices, out_degree)
    drivers = tuple(
        DriverQC(
            driver=identifier,
            target_count=count,
            betweenness=None if scores is None else float(scores[idx]) if idx is not None else 0.0,
            gene_symbol=symbols.get(identifier),
            incoming_count=int(in_degree[idx]) if idx is not None else 0,
        )
        for identifier, count, idx in zip(requested, targets, driver_ids, strict=True)
    )
    if node_count:
        component_count, labels = connected_components(graph, directed=True, connection="weak")
        largest_fraction = float(np.bincount(labels).max() / node_count)
    else:
        component_count = 0
        largest_fraction = None
    q1, median, q3 = np.quantile(targets, [0.25, 0.5, 0.75])
    summary = {
        "nodes": node_count,
        "isolated_nodes": isolated_nodes,
        "edges": graph.nnz,
        "unique_targets": int(np.count_nonzero(in_degree)),
        "requested_drivers": len(requested),
        "active_drivers": active_drivers,
        "zero_target_drivers": len(requested) - active_drivers,
        "active_driver_fraction": active_drivers / len(requested),
        "mean_targets": sum(targets) / len(requested),
        "median_targets": float(median),
        "q1_targets": float(q1),
        "q3_targets": float(q3),
        "weak_components": component_count,
        "largest_weak_component_fraction": largest_fraction,
        "scale_free_adjusted_r2": adjusted_r2,
        "scale_free_slope": slope,
        "betweenness_method": method,
        "betweenness_eligible_sources": eligible_sources,
    }
    return NetworkQC(summary, drivers, degree_fit)

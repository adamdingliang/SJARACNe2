# Network QC

Network QC reads the final directed consensus edge list, requested hub list,
and validated expression input. It applies the same calculations to TF or
SIG networks from bulk, single-cell, spatial, or other expression studies.

From the source checkout, install the optional QC dependencies with
`uv sync --locked --extra qc` or `python -m pip install '.[qc]'`. Use the same
Python version as your workflow environment. For CWL, also select `--extra cwl`
with uv or install `'.[cwl,qc]'` with pip. Add `--network-qc` to the existing
`sjaracne local` command to enable the final task in either engine.

The workflow publishes a self-contained `network_qc.html` plus
`network_qc_summary.tsv` and `network_qc_drivers.tsv`. The report header shows
the SJARACNe2 version and the date, time, and time zone of report generation.
The summary links directly to each of the five sections. Use the **Dark theme**
button below the Summary box to switch the report and its charts to dark mode.

1. **Basic network metrics** start with the configured network type, TF or SIG.
   Runs with no type selected show "Unspecified". The next row gives counts of
   genes with at least one edge and isolated genes. An isolated gene appears in the
   validated expression input and has neither an incoming nor an outgoing
   edge in the final network. The table also shows directed edges, driver
   counts with and without targets, and weak components among the genes with
   edges. The percentage in the largest component uses genes with at least one
   edge and ignores edge direction. The mean and median target counts include
   zero-target drivers.
2. **Target size distribution** plots outgoing target counts for the full
   requested list, including zero-target drivers.
3. **Scale-free fit** uses logarithmic spacing on both axes. Degrees 1, 10,
   and 100 are equally spaced on the x-axis. The fit regresses
   `log(degree frequency + 1/N)` against `log(total degree)` for each degree
   from 1 through the maximum. Total degree is the sum of incoming and
   outgoing edges, and `N` counts genes with at least one edge. Adding
   `1/N` lets the fit include degrees with zero observed frequency.
   The y-axis uses scientific notation for ticks of 0.001 or smaller.
   Adjusted R² summarizes how closely the points follow the line.
   A fit is unavailable for degenerate distributions.
4. **Betweenness centrality** uses exact shortest paths from every node with
   at least one outgoing edge. Paths follow edge direction and each edge has
   equal length. For each ordered pair of other nodes, the score counts the
   share of shortest paths through the driver. The sum is divided by
   `(N - 1) * (N - 2)`, with `N` the number of nodes with at least one edge.
   The report shows the number of starting nodes and uses scientific notation
   for small positive scores.
5. **Drivers** lists every requested driver with its isoformId, geneSymbol,
   incoming edges, outgoing targets, total edges, and normalized betweenness.
   Total edges add incoming and outgoing edges. The report looks up gene
   symbols by isoformId in the expression input. Search by a full or partial
   isoformId or geneSymbol. Click a column header to sort the results, and click
   again to reverse the order. The table shows 20 drivers per page.

The report embeds one copy of Plotly.js and uses aggregated plot data.
The camera button exports the square scale-free plot as a 2,880 × 2,880 PNG
and the other charts as 2,880 × 1,200 PNGs. WDL and CWL use the same Python
report generator.

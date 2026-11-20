"""Create a self-contained HTML report for a final SJARACNe network."""

import argparse
import csv
from datetime import datetime
import json
from html import escape
from importlib import metadata
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from .network_qc_metrics import DriverQC, NetworkQC, compute_network_qc


PLOT_HEIGHT = 400
PLOT_EXPORT_WIDTH = 960

REPORT_SECTIONS = (
    ("basic-network-metrics", "Basic network metrics"),
    ("target-size-distribution", "Target size distribution"),
    ("scale-free-fit", "Scale-free fit"),
    ("betweenness-centrality-distribution", "Betweenness centrality distribution"),
    ("drivers", "Drivers"),
)

SUMMARY_LABELS = {
    "network_type": "Network type",
    "nodes": "Nodes incident to an edge : isolated nodes",
    "edges": "Directed edges",
    "active_drivers": "Drivers with targets : drivers with zero targets",
    "mean_targets": "Mean targets per driver",
    "median_targets": "Median targets per driver",
}


def _display(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _finish_figure(figure):
    figure.update_layout(
        template="plotly_white",
        showlegend=False,
        height=PLOT_HEIGHT,
        margin={"t": 12, "b": 60, "l": 65, "r": 22},
        font={"family": "system-ui, sans-serif", "size": 14},
    )
    return figure


def _empty_figure(message):
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5,
                          xref="paper", yref="paper")
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _finish_figure(figure)


def _target_distribution(counts, median):
    positive = counts[counts > 0]
    if len(positive):
        width = max(1, int(np.ceil(positive.max() / 40)))
        upper = width * int(np.ceil(positive.max() / width)) + 0.5
        edges = np.arange(0.5, upper + 0.5, width)
        labels = [f"{left + .5:.0f}–{right - .5:.0f}"
                  if width > 1 else f"{left + .5:.0f}"
                  for left, right in zip(edges[:-1], edges[1:])]
    else:
        width, edges, labels = 1, np.array([0.5, 1.5]), ["1"]

    bins, _ = np.histogram(positive, bins=edges)
    figure = _zero_histogram(
        counts, bins, edges, labels, "Targets per driver", zero_axis="Zero targets",
    )
    if len(positive) > 1 and np.ptp(positive) > 0:
        x = np.linspace(edges[0], edges[-1], 180)
        # Match the histogram's units, percent of all requested drivers per bin.
        density = gaussian_kde(positive)(x) * width * len(positive) / len(counts) * 100
        figure.add_trace(
            go.Scatter(x=x.tolist(), y=density.tolist(), mode="lines",
                       line={"color": "#146b67", "width": 2},
                       hovertemplate="Smoothed density: %{y:.2f}% per bin<extra></extra>"),
            row=1, col=2,
        )
    if median is not None:
        median_color = "#b4232a"
        if median > 0:
            figure.add_vline(x=median, line_dash="dash", line_color=median_color,
                             line_width=2, row=1, col=2)
        else:
            figure.add_annotation(x="0", y=1.01, yref="y domain", text="Median: 0",
                                  showarrow=False, font_color=median_color, row=1, col=1)
    return figure


def _zero_histogram(values, bins, edges, labels, positive_axis, *, zero_axis="Zero"):
    """Keep the zero mass visible beside the histogram of positive values."""
    num_zero = int(np.count_nonzero(values == 0))
    figure = make_subplots(rows=1, cols=2, column_widths=[0.16, 0.84],
                           horizontal_spacing=0.10, shared_yaxes=True)
    figure.add_trace(
        go.Bar(x=["0"], y=[num_zero * 100 / len(values)], width=[0.55],
               marker_color="#d18242", customdata=[num_zero],
               hovertemplate="%{customdata:,} drivers at zero (%{y:.2f}%)<extra></extra>"),
        row=1, col=1,
    )
    figure.add_trace(
        go.Bar(x=((edges[:-1] + edges[1:]) / 2).tolist(),
               y=(bins * 100 / len(values)).tolist(),
               width=(np.diff(edges) * 0.92).tolist(),
               marker_color="#3e7399", customdata=list(zip(labels, bins.tolist())),
               hovertemplate="%{customdata[0]}<br>%{customdata[1]:,} drivers (%{y:.2f}%)<extra></extra>"),
        row=1, col=2,
    )
    figure.update_xaxes(title_text=zero_axis, row=1, col=1)
    figure.update_xaxes(title_text=positive_axis, row=1, col=2)
    figure.update_yaxes(title_text="Drivers (%)", rangemode="tozero", row=1, col=1)
    return _finish_figure(figure)


def _degree_plot(points):
    if not points:
        return _empty_figure("No positive node degrees to plot.")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=[point.degree for point in points],
                   y=[point.fit_frequency for point in points],
                   mode="markers", marker={"size": 5, "color": "#3e7399"},
                   name="Degree frequency + 1/N",
                   customdata=[point.frequency for point in points],
                   hovertemplate="Degree %{x}<br>Observed frequency %{customdata:.4g}"
                                 "<br>Frequency + 1/N %{y:.4g}<extra></extra>"),
    )
    if all(point.fitted_frequency is not None for point in points):
        figure.add_trace(
            go.Scatter(x=[point.degree for point in points],
                       y=[point.fitted_frequency for point in points],
                       mode="lines", line={"color": "#d18242", "width": 2},
                       name="Fitted line", hoverinfo="skip"),
        )
    figure.update_xaxes(type="log", tickformat="~g", title_text="Total degree per node")
    # Plotly uses scientific notation on decade ticks at 1e-3 and below
    # with exponentformat="e", and decimals on the larger frequency ticks.
    figure.update_yaxes(type="log", dtick=1, exponentformat="e", title_text="Degree frequency + 1/N")
    return _finish_figure(figure)


def _betweenness_distribution(values):
    if not len(values):
        return _empty_figure("Betweenness scores are unavailable for this network.")
    positive = values[values > 0]
    if len(positive):
        smallest, largest = float(positive.min()), float(positive.max())
        if smallest == largest:
            edges = np.array([smallest * 0.9, largest * 1.1])
        elif largest / smallest >= 100:
            edges = np.geomspace(smallest, largest, num=31)
        else:
            edges = np.linspace(smallest, largest, num=31)
        log_scale = largest / smallest >= 100
    else:
        edges, log_scale = np.array([0, 1.0]), False
    bins, _ = np.histogram(positive, bins=edges)
    labels = [f"{left:.3g}–{right:.3g}"
              for left, right in zip(edges[:-1], edges[1:])]
    figure = _zero_histogram(
        values, bins, edges, labels, "Positive score", zero_axis="Zero betweenness",
    )
    if log_scale:
        figure.update_xaxes(type="log", dtick=1, exponentformat="e", row=1, col=2)
    elif len(positive) and largest <= 0.001:
        figure.update_xaxes(tickformat=".2~e", row=1, col=2)
    else:
        figure.update_xaxes(tickformat="~g", row=1, col=2)
    return figure


def _chart_html(figure, chart_id, include_js=False, *, square=False):
    if square:
        # Equal total margins leave the plot area square inside the square chart.
        figure.update_layout(
            autosize=True, height=None, margin={"t": 36, "b": 60, "l": 80, "r": 16},
        )
    html = to_html(
        figure, full_html=False, include_plotlyjs=include_js, include_mathjax=False,
        div_id=chart_id,
        config={
            "responsive": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "width": PLOT_EXPORT_WIDTH,
                "height": PLOT_EXPORT_WIDTH if square else PLOT_HEIGHT,
                "scale": 3,
            },
        },
    )
    return f'<div class="square-chart">{html}</div>' if square else html


def _metric_help(popover_id: str, label: str, message_html: str) -> str:
    return (
        f'<button class="metric-help" type="button" popovertarget="{popover_id}" '
        f'aria-label="{label}" aria-describedby="{popover_id}">'
        '<span aria-hidden="true">?</span></button>'
        f'<span class="metric-help-message" id="{popover_id}" popover>{message_html}</span>'
    )


_DRIVER_TABLE_SCRIPT = """
<script>
(() => {
    const table = document.getElementById("driver-table");
    const drivers = JSON.parse(document.getElementById("driver-data").textContent);
    const headers = Array.from(table.tHead.rows[0].cells);
    const previous = document.getElementById("driver-previous");
    const next = document.getElementById("driver-next");
    const status = document.getElementById("driver-page-status");
    const search = document.getElementById("driver-search");
    const numbers = new Intl.NumberFormat("en-US");
    const names = new Intl.Collator("en", {numeric: true});
    const pageSize = 20;
    let matches = drivers;
    let sortColumn = 0;
    let ascending = true;
    let page = 0;

    function formatCell(value, column) {
        if (value === null) return "\\u2014";
        if (column === 5) {
            return value > 0 && value <= 0.001 ? value.toExponential(4) : String(Number(value.toPrecision(5)));
        }
        return column < 2 ? value : numbers.format(value);
    }

    function showPage() {
        const start = page * pageSize;
        const end = Math.min(start + pageSize, matches.length);
        const body = document.createDocumentFragment();
        for (const driver of matches.slice(start, end)) {
            const row = document.createElement("tr");
            driver.forEach((value, column) => {
                const cell = row.insertCell();
                cell.textContent = formatCell(value, column);
            });
            body.append(row);
        }
        table.tBodies[0].replaceChildren(body);
        status.textContent = matches.length
            ? numbers.format(start + 1) + "\\u2013" + numbers.format(end) + " of " + numbers.format(matches.length)
            : "No matching drivers";
        previous.disabled = page === 0;
        next.disabled = end === matches.length;
    }

    function applySearch() {
        const query = search.value.trim().toLowerCase();
        matches = query
            ? drivers.filter(([identifier, symbol]) =>
                identifier.toLowerCase().includes(query) || (symbol ?? "").toLowerCase().includes(query))
            : drivers;
        page = 0;
        showPage();
    }

    function sortDrivers() {
        drivers.sort((a, b) => {
            const left = a[sortColumn];
            const right = b[sortColumn];
            if (left === right) return names.compare(a[0], b[0]);
            if (left === null) return 1;
            if (right === null) return -1;
            const order = sortColumn < 2 ? names.compare(left, right) : left - right;
            return (ascending ? order : -order) || names.compare(a[0], b[0]);
        });
        for (const [column, header] of headers.entries()) {
            const indicator = header.querySelector(".sort-indicator");
            if (column === sortColumn) {
                header.setAttribute("aria-sort", ascending ? "ascending" : "descending");
                indicator.textContent = ascending ? "\\u2191" : "\\u2193";
            } else {
                header.removeAttribute("aria-sort");
                indicator.textContent = "\\u2195";
            }
        }
        applySearch();
    }

    headers.forEach((header, column) => {
        header.querySelector("button").addEventListener("click", () => {
            ascending = sortColumn === column ? !ascending : column < 2;
            sortColumn = column;
            sortDrivers();
        });
    });
    search.addEventListener("input", applySearch);
    previous.addEventListener("click", () => { page -= 1; showPage(); });
    next.addEventListener("click", () => { page += 1; showPage(); });
    sortDrivers();
})();
</script>
"""


_THEME_SCRIPT = r"""
<script id="report-theme-script">
(() => {
    const root = document.documentElement;
    const toggle = document.getElementById("theme-toggle");
    const plotColors = [
        ["#3e7399", "#83bce7"],
        ["#d18242", "#f1ab6e"],
        ["#146b67", "#6bd1c2"],
        ["#b4232a", "#ff7e85"],
    ];
    let updating = false;
    toggle.addEventListener("click", async () => {
        if (updating) return;
        updating = true;
        try {
            const dark = root.dataset.theme !== "dark";
            root.dataset.theme = dark ? "dark" : "light";
            toggle.textContent = dark ? "Light theme" : "Dark theme";
            const styles = getComputedStyle(root);
            // A null layout value restores the light figure's template default.
            const color = name => dark ? styles.getPropertyValue(name).trim() : null;
            const palette = new Map(plotColors.map(([light, night]) => dark ? [light, night] : [night, light]));
            const plots = document.querySelectorAll(".js-plotly-plot");
            await Promise.all(Array.from(plots, plot => {
                const layout = {
                    paper_bgcolor: color("--surface"),
                    plot_bgcolor: color("--surface"),
                    "font.color": color("--text"),
                    "modebar.bgcolor": color("--surface"),
                    "modebar.color": color("--muted"),
                    "modebar.activecolor": color("--accent"),
                };
                for (const axis of Object.keys(plot.layout).filter(key => /^[xy]axis\d*$/.test(key))) {
                    layout[axis + ".gridcolor"] = color("--divider");
                    layout[axis + ".zerolinecolor"] = color("--divider");
                    layout[axis + ".linecolor"] = color("--divider");
                }
                for (const [i, shape] of (plot.layout.shapes ?? []).entries()) {
                    const newColor = palette.get(shape.line?.color);
                    if (newColor) layout["shapes[" + i + "].line.color"] = newColor;
                }
                for (const [i, annotation] of (plot.layout.annotations ?? []).entries()) {
                    const newColor = palette.get(annotation.font?.color);
                    if (newColor) layout["annotations[" + i + "].font.color"] = newColor;
                }
                const traces = {
                    "marker.color": plot.data.map(trace => palette.get(trace.marker?.color)),
                    "line.color": plot.data.map(trace => palette.get(trace.line?.color)),
                };
                return Plotly.update(plot, traces, layout);
            }));
        } finally {
            updating = false;
        }
    });
})();
</script>
"""


def _driver_table(drivers: tuple[DriverQC, ...]) -> str:
    columns = (
        (("isoformId",), ("isoform", "Id")),
        (("geneSymbol",), ("gene", "Symbol")),
        (("Incoming", "edges"), ("In", "edges")),
        (("Outgoing", "targets"), ("Out", "targets")),
        (("Total", "edges"), ()),
        (("Normalized", "betweenness"), ()),
    )
    headers = []
    for lines, compact_lines in columns:
        label = "<br> ".join(map(escape, lines))
        if compact_lines:
            compact_label = "<br> ".join(map(escape, compact_lines))
            label = (
                f'<span class="driver-label-full">{label}</span>'
                f'<span class="driver-label-compact">{compact_label}</span>'
            )
        headers.append(
            f'<th scope="col"><button type="button" aria-label="{escape(" ".join(lines))}">{label}'
            '<span class="sort-indicator" aria-hidden="true"></span></button></th>'
        )
    data = json.dumps(
        [[row.driver, row.gene_symbol, row.incoming_count, row.target_count,
          row.total_count, row.betweenness] for row in drivers],
        separators=(",", ":"), allow_nan=False,
    ).replace("<", r"\u003c")
    return f"""<section id="drivers">
<h2>Drivers</h2>
<p>Incoming edges point to the driver. Outgoing targets receive an edge from it.
Total edges are the sum of the two counts.</p>
<label class="driver-search" for="driver-search">Search drivers
<input type="search" id="driver-search" placeholder="isoformId or geneSymbol" autocomplete="off"></label>
<div class="driver-table-controls">
<span id="driver-page-status" role="status"></span>
<div><button type="button" id="driver-previous" disabled>Previous</button>
<button type="button" id="driver-next" disabled>Next</button></div>
</div>
<div class="driver-table-scroll" role="region" aria-label="Sortable driver table" tabindex="0">
<table id="driver-table">
<thead><tr>{"".join(headers)}</tr></thead>
<tbody></tbody>
</table>
</div>
<script id="driver-data" type="application/json">{data}</script>
{_DRIVER_TABLE_SCRIPT}
</section>"""


def _report_version() -> str:
    checkout_root = Path(__file__).resolve().parent.parent
    version_file = checkout_root / "version.py"
    # Use this checkout's version even if another release is installed.
    if (checkout_root / "pyproject.toml").is_file() and version_file.is_file():
        return run_path(str(version_file))["__version__"]
    return metadata.version("SJARACNe")


def render_network_qc(report: NetworkQC, *, network_type: str | None = None) -> str:
    """Render summary metrics and grouped plots as an offline HTML page.

    Args:
        report: Computed network statistics and per-driver measurements.
        network_type: Network type selected for inference, when specified.

    Returns:
        A complete HTML document with one embedded Plotly JavaScript bundle.
    """
    summary = report.summary | {"network_type": network_type or "Unspecified"}
    separator = ' <span class="metric-separator">:</span> '
    labels = {key: escape(label).replace(" : ", separator) for key, label in SUMMARY_LABELS.items()}
    values = {key: escape(_display(summary[key])) for key in SUMMARY_LABELS}
    values["nodes"] += separator + escape(_display(summary.get("isolated_nodes")))
    values["active_drivers"] += separator + escape(_display(summary["zero_target_drivers"]))
    driver_count = escape(_display(summary["requested_drivers"]))
    help_buttons = {
        "edges": _metric_help(
            "directed-edges-help", "How directed edges are counted",
            "If both <code>A &rarr; B</code> and <code>B &rarr; A</code> are present, they count as two edges.",
        ),
        "mean_targets": _metric_help(
            "target-count-help", "Which drivers are included in the mean and median",
            f"The mean and median include all <strong>{driver_count} requested drivers</strong>.",
        ),
    }
    rows = "".join(
        f'<tr><th scope="row">{label}{help_buttons.get(key, "")}</th>'
        f'<td>{values[key]}</td></tr>'
        for key, label in labels.items()
    )
    counts = np.array([driver.target_count for driver in report.drivers], dtype=int)
    scores = np.array([driver.betweenness for driver in report.drivers
                       if driver.betweenness is not None], dtype=float)
    target_plot = (_target_distribution(counts, summary["median_targets"]) if len(counts)
                   else _empty_figure("No requested drivers."))
    target_html = _chart_html(target_plot, "target-size-chart", include_js=True)
    degree_html = _chart_html(_degree_plot(report.degree_fit), "degree-fit-chart", square=True)
    betweenness_html = _chart_html(_betweenness_distribution(scores), "betweenness-chart")
    if summary["betweenness_method"] == "exact":
        method_note = (
            f'Exact calculation across all {summary["betweenness_eligible_sources"]:,} eligible source nodes.'
        )
    else:
        method_note = "The graph is too small for normalized betweenness."
    betweenness_help = _metric_help(
        "betweenness-help", "How normalized betweenness is calculated",
        "<strong>Normalized betweenness</strong> sums the share of shortest directed paths "
        "through the driver for each ordered pair of other nodes. We divide this sum by "
        "<code>(N - 1)(N - 2)</code>, where N counts nodes with at least one edge.<br><br>"
        "<strong>Eligible source nodes</strong> have at least one outgoing edge, "
        "so they can start a directed path.",
    )
    median = escape(_display(summary["median_targets"]))
    slope = escape(_display(summary["scale_free_slope"]))
    r2 = escape(_display(summary["scale_free_adjusted_r2"]))
    driver_table = _driver_table(report.drivers)
    degree_fit_help = _metric_help(
        "degree-fit-help", "What adjusted R2 measures",
        "Adjusted R2 summarizes how closely the points follow the line.",
    )
    generated_at = datetime.now().astimezone()
    timestamp = generated_at.isoformat(timespec="seconds")
    display_time = generated_at.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    report_version = escape(_report_version())
    summary_links = "".join(
        f'<li><a href="#{anchor}">{escape(title)}</a></li>' for anchor, title in REPORT_SECTIONS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SJARACNe2 Network QC Report</title>
<style>
:root {{
    color-scheme: light;
    --page: #f4f7f9; --surface: #fff; --text: #162332;
    --accent: #224d69; --muted: #36536c; --border: #dce4eb; --divider: #d7e2eb;
    --table-heading: #e5edf4; --row-alt: #f3f7fa; --row-hover: #e1edf7;
    --separator: #a04a2a; --icon-border: #658099; --control-border: #a9bccc;
    --control-hover: #d8e6f0; --disabled: #697782;
}}
:root[data-theme="dark"] {{
    color-scheme: dark;
    --page: #101820; --surface: #18232f; --text: #e5edf5;
    --accent: #a9d3f5; --muted: #b3c4d2; --border: #35485a; --divider: #394e61;
    --table-heading: #26384a; --row-alt: #1e2c3a; --row-hover: #283d51;
    --separator: #f0ab83; --icon-border: #799ab3; --control-border: #6b8296;
    --control-hover: #31475c; --disabled: #91a4b5;
}}
body {{ margin: 0; background: var(--page); color: var(--text); font: 16px/1.5 system-ui, sans-serif; }}
.report-header {{ background: var(--surface); border-bottom: 1px solid var(--border); }}
.report-header-inner, .report-layout {{ width: min(76rem, 94vw); margin-inline: auto; }}
.report-header-inner {{ display: flex; justify-content: space-between; align-items: flex-start;
                        gap: 1rem; padding-block: 1.1rem; }}
.report-header h1 {{ margin: 0; font-size: clamp(1.35rem, 3.3vw, 1.95rem); }}
.report-meta {{ text-align: right; color: var(--muted); font-size: .9rem; }}
.report-meta span, .report-meta time {{ display: block; white-space: nowrap; }}
.report-meta time {{ font-variant-numeric: tabular-nums; }}
.report-layout {{ display: grid; grid-template-columns: 12rem minmax(0, 1fr);
                  gap: 1.25rem; align-items: start; margin-block: 1.25rem 2rem; }}
.report-sidebar {{ position: sticky; top: 1rem; }}
.report-summary {{ padding: .8rem; border-radius: 9px;
                   background: var(--surface); border: 1px solid var(--border); font-size: .93rem; }}
.report-summary .summary-title {{ margin: 0; font-weight: 700; }}
.report-summary ul {{ list-style: none; padding: 0; margin: .4rem 0 0; }}
.report-summary a {{ display: block; color: var(--accent); padding: .3rem .35rem;
                    border-radius: 4px; text-underline-offset: .15em; }}
.report-summary a:hover, .report-summary a:focus-visible {{ background: var(--row-hover); }}
main {{ min-width: 0; }}
section {{ margin: 0 0 1rem; padding: clamp(1rem, 3vw, 1.75rem); background: var(--surface);
           border: 1px solid var(--border); border-radius: 9px; scroll-margin-top: 1rem; }}
h1, h2 {{ margin: 0 0 .6em; line-height: 1.2; }}
@media (max-width: 850px) {{
    .report-layout {{ grid-template-columns: minmax(0, 1fr); gap: .9rem; margin-block-start: .9rem; }}
    .report-sidebar {{ position: static; }}
    .report-summary ul {{ display: flex; flex-wrap: wrap; gap: .15rem .45rem; }}
}}
@media (max-width: 540px) {{
    .report-header-inner {{ flex-direction: column; gap: .5rem; }}
    .report-meta {{ align-self: flex-end; }}
}}
p {{ margin: .6em 0; }}
.square-chart {{ position: relative; width: min(100%, 40rem); aspect-ratio: 1; margin-inline: auto; }}
.square-chart > div {{ position: absolute; inset: 0; }}
.metrics {{ width: 100%; max-width: 42rem; border-collapse: collapse; text-align: left; }}
.metrics th, .metrics td {{ padding: .45em .7em; border-bottom: 1px solid var(--divider); }}
.metrics thead {{ background: var(--table-heading); }}
.metrics tbody tr:nth-child(even) {{ background: var(--row-alt); }}
.metrics tbody tr:hover, .metrics tbody tr:focus-within {{ background: var(--row-hover); }}
.metrics th:last-child, .metrics td {{ text-align: right; }}
.metrics td {{ border-inline-start: 1px solid var(--divider); color: var(--accent);
               font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.metric-separator {{ color: var(--separator); font-weight: 800; padding-inline: .12em; }}
.metric-help {{ display: inline-grid; place-items: center; vertical-align: middle;
                width: 1.5rem; height: 1.5rem; margin-inline-start: .15rem; padding: 0;
                border: 0; border-radius: 50%; background: transparent; cursor: pointer; }}
.metric-help span {{ display: grid; place-items: center; box-sizing: border-box;
                     width: 1rem; height: 1rem; border: 1px solid var(--icon-border); border-radius: 50%;
                     background: var(--surface); color: var(--accent); font: bold .6875rem/1 system-ui, sans-serif; }}
.metric-help:hover span {{ background: var(--control-hover); }}
.metric-help:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
.driver-search {{ display: grid; width: min(22rem, 100%); gap: .25rem; margin-block-start: .8rem;
                  font-weight: 600; }}
.driver-search input {{ box-sizing: border-box; width: 100%; padding: .4rem .55rem;
                        border: 1px solid var(--control-border); border-radius: 5px;
                        color: inherit; background: var(--surface); font: inherit; font-weight: 400; }}
.driver-search input:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
.driver-table-controls {{ display: flex; flex-wrap: wrap; justify-content: space-between;
                          align-items: center; gap: .5rem; margin: .8rem 0 .45rem; }}
#theme-toggle, .driver-table-controls button {{ border: 1px solid var(--control-border); border-radius: 5px; background: var(--page);
                                 color: var(--accent); padding: .3rem .6rem; font: inherit; cursor: pointer; }}
#theme-toggle {{ margin-block-start: .75rem; }}
#theme-toggle:hover {{ background: var(--control-hover); }}
#theme-toggle:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
.driver-table-controls button:disabled {{ color: var(--disabled); cursor: default; }}
.driver-table-scroll {{ overflow-x: auto; }}
#driver-table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: .9rem; }}
#driver-table th, #driver-table td {{ padding: .38rem .42rem; border-bottom: 1px solid var(--divider);
                                     overflow-wrap: anywhere; }}
#driver-table th {{ padding: 0; background: var(--table-heading); }}
#driver-table th button {{ display: block; border: 0; background: none; width: 100%;
                           text-align: inherit; padding: .4rem .42rem; color: inherit;
                           cursor: pointer; font: inherit; font-weight: 700; line-height: 1.2;
                           overflow-wrap: anywhere; }}
#driver-table th button:hover {{ background: var(--control-hover); }}
#driver-table th button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
#driver-table td:nth-child(n+3), #driver-table th:nth-child(n+3) {{
    text-align: right; font-variant-numeric: tabular-nums; }}
#driver-table td:nth-child(n+3) {{ white-space: nowrap; }}
#driver-table tbody tr:nth-child(even) {{ background: var(--row-alt); }}
#driver-table tbody tr:hover {{ background: var(--row-hover); }}
.sort-indicator {{ display: inline-block; margin-inline-start: .15em; color: var(--muted); }}
.driver-label-compact {{ display: none; }}
@media (max-width: 540px) {{
    #driver-table {{ font-size: .75rem; }}
    #driver-table td {{ padding: .3rem .25rem; }}
    #driver-table th button {{ padding: .3rem .125rem; font-size: .6875rem; white-space: nowrap; }}
    #driver-table th:not([aria-sort]) .sort-indicator, .driver-label-full {{ display: none; }}
    .driver-label-compact {{ display: inline; }}
}}
@media (max-width: 350px) {{
    #drivers {{ padding-inline: .5rem; }}
}}
.metric-help-message {{ box-sizing: border-box; width: min(17rem, calc(100vw - 2rem));
                        padding: .6rem .75rem; border: 1px solid var(--control-border); border-radius: 6px;
                        background: var(--surface); color: var(--text); font-weight: 400;
                        box-shadow: 0 3px 12px rgb(22 35 50 / 15%); }}
@supports (position-area: block-end span-all) {{
    .metric-help-message {{ inset: auto; margin: .35rem 0; position-area: block-end span-all;
                            justify-self: anchor-center; position-try-fallbacks: flip-block; }}
}}
@supports not selector(:popover-open) {{
    .metric-help {{ display: none; }}
    .metric-help-message {{ display: block; width: auto; }}
}}
</style>
</head>
<body>
<header class="report-header">
<div class="report-header-inner">
<h1>SJARACNe2 Network QC Report</h1>
<div class="report-meta">
<span>SJARACNe2 v{report_version}</span>
<time datetime="{timestamp}">{escape(display_time)}</time>
</div>
</div>
</header>
<div class="report-layout">
<aside class="report-sidebar">
<nav class="report-summary" id="report-summary" aria-label="Report summary">
<p class="summary-title">Summary</p>
<ul>{summary_links}</ul>
</nav>
<button type="button" id="theme-toggle">Dark theme</button>
</aside>
<main>
<section id="basic-network-metrics">
<h2>Basic network metrics</h2>
<table class="metrics">
<thead><tr><th scope="col">Metric</th><th scope="col">Value</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</section>
<section id="target-size-distribution">
<h2>Target size distribution</h2>
<p>Target counts use outgoing edges for all requested drivers. Median targets per driver
<strong>{median}</strong>. The orange bar shows the zero-target fraction.</p>
{target_html}
</section>
<section id="scale-free-fit">
<h2>Scale-free fit</h2>
<p>Both axes use logarithmic spacing. For example, degrees 1, 10, and 100 are
equally spaced on the x-axis. The line fits <code>log(degree frequency + 1/N)</code>
against <code>log(total degree)</code>, where N is the number of nodes with at least one edge.</p>
<p><strong>Adjusted R2: {r2}; Slope: {slope}</strong>{degree_fit_help}</p>
{degree_html}
</section>
<section id="betweenness-centrality-distribution">
<h2>Betweenness centrality distribution</h2>
<p>Normalized betweenness of requested drivers using directed, unweighted shortest paths.
{escape(method_note)}{betweenness_help}</p>
{betweenness_html}
</section>
{driver_table}
</main>
</div>
{_THEME_SCRIPT}
</body></html>
"""


def _write_table(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(("NA" if value is None else value for value in row)
                         for row in rows)


def _expression_gene_count(validation_report: Path) -> int:
    # The preflight report already records the number of unique expression IDs.
    with validation_report.open(encoding="utf-8") as handle:
        fields = dict(line.rstrip("\n").split("\t", 1) for line in handle if "\t" in line)
    try:
        count = int(fields.get("expression_genes", ""))
    except ValueError as error:
        raise ValueError(f"Invalid expression gene count in {validation_report}") from error
    if fields.get("status") != "passed" or count < 1:
        raise ValueError(f"Invalid input validation report: {validation_report}")
    return count


def write_network_qc(
    network_file: Path,
    hubs_file: Path,
    output_dir: Path,
    *,
    validation_report: Path | None = None,
    network_type: str | None = None,
    expression_file: Path | None = None,
    replace_existing: bool = True,
) -> None:
    """Write HTML and TSV metrics for an existing network.

    Args:
        network_file: Final consensus edge-list TSV.
        hubs_file: Original requested driver identifiers, one per line.
        output_dir: Directory in which to write the three QC artifacts.
        validation_report: Input validation report providing the expression gene count.
        network_type: Network type selected for inference, when specified.
        expression_file: Expression TSV providing driver symbols and, when
            needed, the universe of expression genes.
        replace_existing: Allow an existing QC artifact to be replaced.

    Raises:
        FileExistsError: A QC destination exists when replacement is disabled.
        OSError: Inputs cannot be read or outputs cannot be written.
        ValueError: An input or validation report is invalid.
    """
    output_names = ("network_qc_summary.tsv", "network_qc_drivers.tsv", "network_qc.html")
    output_dir = Path(output_dir)
    if not replace_existing:
        for name in output_names:
            destination = output_dir / name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"QC output already exists: {destination}")

    count = _expression_gene_count(validation_report) if validation_report is not None else None
    report = compute_network_qc(
        network_file, hubs_file, expression_gene_count=count, expression_file=expression_file,
    )
    html = render_network_qc(report, network_type=network_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".network-qc-", dir=output_dir) as temporary:
        stage = Path(temporary)
        _write_table(stage / "network_qc_summary.tsv", ("metric", "value"),
                     report.summary.items())
        _write_table(
            stage / "network_qc_drivers.tsv",
            ("driver", "target_count", "betweenness", "geneSymbol", "incoming_edges", "total_edges"),
            ((row.driver, row.target_count, row.betweenness, row.gene_symbol,
              row.incoming_count, row.total_count) for row in report.drivers),
        )
        (stage / "network_qc.html").write_text(html, encoding="utf-8", newline="\n")
        if replace_existing:
            for name in output_names:
                (stage / name).replace(output_dir / name)
            return

        published = []
        try:
            for name in output_names:
                destination = output_dir / name
                # The stage is on the same filesystem. Linking publishes a
                # complete file and fails if a concurrent run claimed the name.
                destination.hardlink_to(stage / name)
                published.append(destination)
        except OSError:
            for destination in published:
                destination.unlink(missing_ok=True)
            raise


def main(argv=None):
    """Run the internal QC workflow task."""
    from .sjaracne import add_network_qc_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_network_qc_arguments(parser)
    args = parser.parse_args(argv)
    try:
        write_network_qc(
            args.network, args.hubs, args.output_dir,
            validation_report=args.validation_report, network_type=args.network_type,
            expression_file=args.expression,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Network QC failed: {error}\n")


if __name__ == "__main__":
    main()

"""Check the report against graphs with known target counts and shortest paths."""

from contextlib import redirect_stderr
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from SJARACNe.network_qc import render_network_qc, write_network_qc
from SJARACNe.network_qc_metrics import compute_network_qc
from SJARACNe import sjaracne


class NetworkQCTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.network = self.root / "network.tsv"
        self.hubs = self.root / "hubs.txt"
        self.hubs.write_text("A\nB\nC\nD\nX\nY\nMISSING\n", encoding="utf-8")

    def measure(self, rows, **options):
        self.network.write_text("source\ttarget\tMI\n" + rows, encoding="utf-8")
        return compute_network_qc(self.network, self.hubs, **options)

    def test_shortest_path_ties_and_zero_target_drivers(self):
        # A has two equally short paths through B/C to D, and on to E.
        # X -> Y is disconnected. Y is requested but only appears as a target.
        expression = self.root / "expression.tsv"
        expression.write_text(
            "isoformId\tgeneSymbol\tS0\nA\tA</script>\t1\nB\tBeta\t2\nD\tDelta\t3\nY\tYpsilon\t4\n",
            encoding="utf-8",
        )
        qc = self.measure(
            "A\tB\t1\nA\tC\t30\nB\tD\t1\nC\tD\t30\nD\tE\t1\nX\tY\t1\n",
            expression_gene_count=10, expression_file=expression,
        )
        self.assertEqual(qc.summary["nodes"], 7)
        self.assertEqual(qc.summary["isolated_nodes"], 3)
        self.assertEqual(qc.summary["edges"], 6)
        self.assertEqual(qc.summary["unique_targets"], 5)
        self.assertEqual(qc.summary["requested_drivers"], 7)
        self.assertEqual(qc.summary["active_drivers"], 5)
        self.assertEqual(qc.summary["zero_target_drivers"], 2)
        self.assertAlmostEqual(qc.summary["mean_targets"], 6 / 7)
        self.assertEqual(qc.summary["median_targets"], 1)
        self.assertEqual(qc.summary["weak_components"], 2)
        self.assertAlmostEqual(qc.summary["largest_weak_component_fraction"], 5 / 7)
        html = render_network_qc(qc)
        self.assertNotIn("Weakly connected components", html)
        self.assertNotIn("Percentage of nodes in the largest component", html)
        self.assertIn('"text":"Zero targets"', html)
        self.assertIn('"color":"#b4232a","dash":"dash"', html)
        self.assertEqual(html.count('"toImageButtonOptions":'), 3)
        self.assertTrue('<div class="square-chart">' in html)
        self.assertIn('"tickformat":"~g"', html)
        self.assertIn('"dtick":1,"exponentformat":"e"', html)
        self.assertIn('"width": 960, "height": 960, "scale": 3', html)
        self.assertTrue("Both axes use logarithmic spacing." in html)
        self.assertTrue("degrees 1, 10, and 100 are\nequally spaced on the x-axis." in html)
        self.assertTrue("log(degree frequency + 1/N)" in html)
        self.assertFalse("NetBID2-style" in html or "This is a descriptive fit" in html)
        self.assertEqual(qc.summary["betweenness_method"], "exact")
        self.assertEqual(qc.summary["betweenness_eligible_sources"], 5)
        self.assertTrue("Exact calculation across all 5 eligible source nodes." in html)
        self.assertTrue('popovertarget="betweenness-help"' in html)
        self.assertTrue("<strong>Normalized betweenness</strong>" in html)
        self.assertTrue("<strong>Eligible source nodes</strong> have at least one outgoing edge" in html)
        self.assertTrue('"text":"Zero betweenness"' in html)
        table_json = html.split('<script id="driver-data" type="application/json">', 1)[1].split("</script>", 1)[0]
        records = {row[0]: row[1:] for row in json.loads(table_json)}
        self.assertEqual(len(records), 7)
        self.assertEqual(records["A"][:4], ["A</script>", 0, 2, 2])
        self.assertEqual(records["D"][:4], ["Delta", 2, 1, 3])
        self.assertAlmostEqual(records["D"][4], 3 / 30)
        self.assertEqual(records["Y"], ["Ypsilon", 1, 0, 1, 0])
        self.assertEqual(records["MISSING"], [None, 0, 0, 0, 0])
        drivers = {d.driver: d for d in qc.drivers}
        self.assertAlmostEqual(drivers["B"].betweenness, 1 / 30)
        self.assertAlmostEqual(drivers["C"].betweenness, 1 / 30)
        self.assertAlmostEqual(drivers["D"].betweenness, 3 / 30)
        for name in ("A", "X", "Y", "MISSING"):
            self.assertEqual(drivers[name].betweenness, 0)
        for name in ("Y", "MISSING"):
            self.assertEqual(drivers[name].target_count, 0)

    def test_exact_scores_use_all_sources_and_are_reproducible(self):
        size = 520
        genes = [f"D{i:03d}" for i in range(size)]
        self.hubs.write_text("\n".join(genes) + "\n", encoding="utf-8")
        rows = [f"{gene}\t{genes[(i + 1) % size]}\t1\n" for i, gene in enumerate(genes)]
        first = self.measure("".join(rows))
        second = self.measure("".join(reversed(rows)))
        self.assertEqual(first.summary["betweenness_method"], "exact")
        self.assertEqual(first.summary["betweenness_eligible_sources"], size)
        self.assertNotIn("betweenness_sample_size", first.summary)
        self.assertNotIn("betweenness_seed", first.summary)
        self.assertEqual(first.drivers, second.drivers)
        # Half of the ordered pairs of other genes pass through each node.
        for driver in first.drivers:
            self.assertAlmostEqual(driver.betweenness, 0.5)

    def test_netbid_degree_fit_keeps_unobserved_bins(self):
        qc = self.measure("A\tB\t1\nA\tC\t1\nA\tD\t1\n")
        self.assertEqual([p.degree for p in qc.degree_fit], [1, 2, 3])
        self.assertEqual([p.frequency for p in qc.degree_fit], [0.75, 0, 0.25])
        self.assertEqual([p.fit_frequency for p in qc.degree_fit], [1.0, 0.25, 0.5])
        self.assertIsNotNone(qc.summary["scale_free_adjusted_r2"])

    def test_header_only_network_writes_all_outputs(self):
        qc = self.measure("")
        self.assertEqual(qc.summary["zero_target_drivers"], 7)
        self.assertEqual(qc.summary["mean_targets"], 0)
        self.assertEqual(qc.summary["median_targets"], 0)
        self.assertIsNone(qc.summary["scale_free_adjusted_r2"])
        self.assertEqual(qc.summary["betweenness_method"], "unavailable")
        output = self.root / "report"
        self.assertIsNone(qc.summary["isolated_nodes"])
        validation = self.root / "validation.txt"
        validation.write_text("status\tpassed\nexpression_genes\t8\n", encoding="utf-8")
        self.assertIn('<tbody><tr><th scope="row">Network type</th><td>Unspecified</td>',
                      render_network_qc(qc))
        write_network_qc(self.network, self.hubs, output, validation_report=validation, network_type="SIG")
        html = (output / "network_qc.html").read_text(encoding="utf-8")
        self.assertIn('<tbody><tr><th scope="row">Network type</th><td>SIG</td>', html)
        self.assertIn('Mean targets per driver<button class="metric-help"', html)
        self.assertIn('popovertarget="target-count-help"', html)
        self.assertIn('The mean and median include all <strong>7 requested drivers</strong>.', html)
        self.assertIn('Median targets per driver</th>', html)
        self.assertTrue('Adjusted R2: Unavailable; Slope: Unavailable</strong><button' in html)
        self.assertTrue('id="degree-fit-help" popover>'
                        'Adjusted R2 summarizes how closely the points follow the line.</span>' in html)
        self.assertNotIn("weak-components-help", html)
        self.assertEqual(html.count("<h2>"), 5)
        self.assertTrue("<h1>SJARACNe2 Network QC Report</h1>" in html)
        self.assertTrue('</nav>\n<button type="button" id="theme-toggle">Dark theme</button>\n</aside>' in html)
        self.assertTrue(':root[data-theme="dark"]' in html)
        self.assertTrue('<script id="report-theme-script">' in html)
        self.assertTrue('<div class="report-meta">\n<span>SJARACNe2 v' in html)
        timestamp = html.split('<time datetime="', 1)[1].split('"', 1)[0]
        self.assertIsNotNone(datetime.fromisoformat(timestamp).utcoffset())
        for anchor, title in (
            ("basic-network-metrics", "Basic network metrics"),
            ("target-size-distribution", "Target size distribution"),
            ("scale-free-fit", "Scale-free fit"),
            ("betweenness-centrality-distribution", "Betweenness centrality distribution"),
            ("drivers", "Drivers"),
        ):
            self.assertTrue(f'<a href="#{anchor}">{title}</a>' in html)
            self.assertTrue(f'<section id="{anchor}">\n<h2>{title}</h2>' in html)
        self.assertTrue('<section id="drivers">' in html)
        self.assertTrue('type="search" id="driver-search" placeholder="isoformId or geneSymbol"' in html)
        self.assertTrue("const pageSize = 20;" in html)
        self.assertTrue('aria-label="Incoming edges"><span class="driver-label-full">Incoming<br> edges' in html)
        self.assertTrue('aria-label="Outgoing targets"><span class="driver-label-full">Outgoing<br> targets' in html)
        self.assertTrue('<span class="driver-label-compact">In<br> edges</span>' in html)
        self.assertTrue('<span class="driver-label-compact">Out<br> targets</span>' in html)
        self.assertEqual(html.count("plotly.js v"), 1)
        self.assertNotIn("<script src=", html)
        separator = ' <span class="metric-separator">:</span> '
        self.assertIn(f'Nodes incident to an edge{separator}isolated nodes</th><td>0{separator}8</td>', html)
        self.assertIn(f'Drivers with targets{separator}drivers with zero targets</th><td>0{separator}7</td>', html)
        self.assertNotIn('>Unique targets</th>', html)
        self.assertNotIn('>Requested drivers</th>', html)
        self.assertNotIn('>Active driver fraction</th>', html)
        self.assertNotIn('>25th percentile of target counts</th>', html)
        self.assertNotIn('>75th percentile of target counts</th>', html)
        self.assertIn('Directed edges<button class="metric-help"', html)
        self.assertIn('popovertarget="directed-edges-help"', html)
        self.assertIn('id="directed-edges-help" popover', html)
        self.assertIn('If both <code>A &rarr; B</code> and <code>B &rarr; A</code> are present, '
                      'they count as two edges.', html)
        with (output / "network_qc_drivers.tsv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0], {
            "driver": "A", "target_count": "0", "betweenness": "NA",
            "geneSymbol": "NA", "incoming_edges": "0", "total_edges": "0",
        })
        summary = (output / "network_qc_summary.tsv").read_text(encoding="utf-8")
        self.assertIn("\nisolated_nodes\t8\n", summary)

    def test_standalone_cli_uses_existing_inputs_and_protects_reports(self):
        self.network.write_text(
            "source\ttarget\tsource.symbol\ttarget.symbol\tMI\n"
            "A\tB\tAlpha\tBeta\t0.3\nB\tC\tBeta\tGamma\t0.4\n",
            encoding="utf-8",
        )
        expression = self.root / "original expression.exp"
        expression.write_text(
            "isoformId\tgeneSymbol\ts1\n"
            "A\tAlpha\t1\nB\tBeta\t2\nC\tGamma\t3\nD\tDelta\t4\n"
            "D\tDuplicate\t4\nX\tExtra\t5\n",
            encoding="utf-8",
        )
        output = self.root / "existing output"
        output.mkdir()
        unrelated = output / "run.json"
        unrelated.write_text("Existing run", encoding="utf-8")
        arguments = [
            "qc", "--network", str(self.network), "--hubs", str(self.hubs),
            "--expression", str(expression), "--network-type", "tf", "-o", str(output),
        ]
        with mock.patch.object(sjaracne, "require_tool", side_effect=AssertionError("Workflow engine used")):
            self.assertEqual(sjaracne.main(arguments), 0)

        files = {path.name: path.read_bytes() for path in output.iterdir()}
        self.assertIn(b"Network type</th><td>TF</td>", files["network_qc.html"])
        self.assertIn(b"\nisolated_nodes\t2\n", files["network_qc_summary.tsv"])
        with (output / "network_qc_drivers.tsv").open(newline="", encoding="utf-8") as handle:
            drivers = {row["driver"]: row for row in csv.DictReader(handle, delimiter="\t")}
        self.assertEqual(drivers["D"]["geneSymbol"], "Delta")
        self.assertEqual(drivers["D"]["target_count"], "0")
        self.assertEqual(files[unrelated.name], b"Existing run")

        diagnostics = io.StringIO()
        with redirect_stderr(diagnostics), self.assertRaises(SystemExit) as error:
            sjaracne.main(arguments)
        self.assertEqual(error.exception.code, 1)
        self.assertIn("QC output already exists", diagnostics.getvalue())
        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, files)

    def test_standalone_expression_must_contain_network_nodes(self):
        expression = self.root / "different.exp"
        expression.write_text(
            "isoformId\tgeneSymbol\ts1\nA\tAlpha\t1\nD\tDelta\t2\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Network ID 'B' is absent"):
            self.measure("A\tB\t1\n", expression_file=expression)

    def test_no_replace_publish_rolls_back_its_own_files_on_collision(self):
        self.network.write_text("source\ttarget\tMI\n", encoding="utf-8")
        output = self.root / "qc"
        original_link = Path.hardlink_to

        def competing_publication(destination, source):
            if destination.name == "network_qc_drivers.tsv":
                destination.write_bytes(b"Other report")
            original_link(destination, source)

        with mock.patch.object(Path, "hardlink_to", competing_publication):
            with self.assertRaises(FileExistsError):
                write_network_qc(self.network, self.hubs, output, replace_existing=False)
        self.assertEqual(
            {path.name: path.read_bytes() for path in output.iterdir()},
            {"network_qc_drivers.tsv": b"Other report"},
        )

    def test_malformed_graph_is_rejected(self):
        for rows, message in (("A\tB\t1\nA\tB\t2\n", "duplicate"),
                              ("A\tA\t1\n", "self-loop"),
                              ("A\tB\tnan\n", "finite")):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, message):
                self.measure(rows)
        with self.assertRaisesRegex(ValueError, "Expression gene count"):
            self.measure("A\tB\t1\n", expression_gene_count=1)

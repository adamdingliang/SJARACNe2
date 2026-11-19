#!/usr/bin/env cwl-runner
cwlVersion: v1.0
class: CommandLineTool
id: main
label: Report network quality control
doc: |
  Generate a report and metric tables from the final network and the full hub
  list, using the prepared expression for driver symbols and validated gene count
  from preflight. CWL 1.0 has no conditional step, so a disabled report runs a
  no-op without staging the expression matrix or loading the report module.
requirements:
  InlineJavascriptRequirement: {}
  ResourceRequirement:
    coresMin: 1
    ramMin: '$(inputs.enabled ? 2048 : 256)'
baseCommand: python3
arguments:
  - position: 1
    valueFrom: |
      ${
        if (!inputs.enabled) {
          return ["-c", "pass"];
        }
        if (inputs.expression == null) {
          throw new Error("Enabled network QC requires the prepared expression matrix");
        }
        var args = ["-m", "SJARACNe.network_qc", "--network", inputs.network.path,
                    "--hubs", inputs.hubs.path,
                    "--expression", inputs.expression.path,
                    "--validation-report", inputs.validation_report.path,
                    "--output-dir", "network-qc"];
        if (inputs.network_type != null) {
          args.push("--network-type", inputs.network_type.toUpperCase());
        }
        return args;
      }
inputs:
  enabled: boolean
  network: File
  hubs: File
  expression: File?
  validation_report: File
  network_type: string?
outputs:
  html:
    type: File?
    outputBinding: {glob: network-qc/network_qc.html}
  summary:
    type: File?
    outputBinding: {glob: network-qc/network_qc_summary.tsv}
  drivers:
    type: File?
    outputBinding: {glob: network-qc/network_qc_drivers.tsv}

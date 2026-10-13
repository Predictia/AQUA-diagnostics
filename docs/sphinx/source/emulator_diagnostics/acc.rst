ACC
===

ACC compares anomalies against an explicitly configured climatology. It returns
pointwise temporal correlations and time-varying spatial pattern correlations.
Set ``diagnostics.acc.climatology``; no model-specific climatology is assumed.
Run ``python -m aqua.diagnostics.emulator.acc.cli_acc --config CONFIG``.

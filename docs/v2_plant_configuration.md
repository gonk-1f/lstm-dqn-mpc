# v2 Plant Configuration

The v2 model keeps the approved research simulation and the real-vessel
specification as separate provenance records. Their values are not
interchangeable.

## Research simulation configuration

| Parameter | Value | Provenance | Status |
|---|---:|---|---|
| Aggregate fuel-cell rated power | 600 kW | Yang et al., *Ocean Engineering* (2026), DOI `10.1016/j.oceaneng.2026.125687` | approved research simulation input |
| Battery nominal energy | 624 kWh | same simulation configuration | approved research simulation input |
| Battery charge lower bound | -624 kW | same source, Table 6 | approved for objective-scale audit |
| Battery discharge upper bound | +1248 kW | same source, Table 6 | approved for objective-scale audit |

These are the values encoded by `PlantConfig.research_simulation()`. The power
bounds were explicitly accepted by the user on 2026-09-22 for the objective-
scale audit. They must
not be described as real-vessel specifications and are not sourced to a JMSE
2025 paper.

## Real-vessel technical specification

Source:
`C:\Users\20883\OneDrive\Desktop\氢舟一号\rightpdf_“三峡氢舟1号”动力系统系统技术规格书 - V1_word2pdf.pdf`
(19 pages; SHA-256
`C269F9D7E9DEDC23118514FED8EBC0987500948ABF88F8A2129EA0F264315A34`).

| Parameter | Specification value |
|---|---:|
| Aggregate fuel-cell rated power | 560 kW |
| Fuel-cell arrangement | 8 modules x 70 kW |
| Battery nominal energy | approximately 1,806 kWh |
| Battery clusters | 12 |
| Rated energy per cluster | 150.5 kWh |
| Battery rated output | at least 900 kW |
| Battery rated voltage | 537.6 V |
| End-of-operation condition | shore charging |

These values are encoded separately by
`RealVesselSpecification.from_authoritative_specification()`. No simulation may
silently combine the 560 kW / approximately 1,806 kWh vessel configuration with
the 600 kW / 624 kWh research configuration.

For the Train-only supervisory audit, the 12 equal-capacity cluster SOC values
may therefore be aggregated by arithmetic mean only after every cluster passes
duplicate handling, causal alignment, and freshness checks. The specification's
"at least 900 kW" output requirement is not a maximum discharge bound and does
not define a charging lower bound.

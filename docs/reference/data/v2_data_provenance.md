# v2 Data Provenance

## Accepted source boundary

The candidate source classes are:

- user-provided Excel workbooks (`.xlsx` and `.xls`) for prospective original
  measurements; and
- an authoritative vessel technical specification.

Candidate acceptance is not a usability decision. Each prospective measurement
must separately record workbook, sheet, column, unit, timestamp field, measured
sampling interval, missing rate, physical meaning, usability, and the reason for
that decision. It must also carry an explicit lineage classification. Only an
Excel record classified as `original_measurement` can be usable; rename-helper,
processed-aggregate, interpolated, generated, and digitized records remain
unusable regardless of filename or otherwise complete metadata. A scanned
workbook therefore enters the inventory as an `unaudited_candidate` until those
facts are explicitly audited.

Historical CSV/MAT data, 1 s interpolation, generated profiles, cleaned
aggregates, and image/digitization transcriptions cannot be promoted to formal
v2 raw facts under the current contract. They may be retained only as derivative
v1 lineage.

## Evidence found in the authoritative user-data root

The 2026-09-14 audit of
`C:\Users\20883\OneDrive\Desktop\氢舟一号` found both Excel files and a
technical specification; earlier claims that neither existed were false and are
superseded by this audit.

The authoritative specification is:
`C:\Users\20883\OneDrive\Desktop\氢舟一号\rightpdf_“三峡氢舟1号”动力系统系统技术规格书 - V1_word2pdf.pdf`.
It has 19 pages and SHA-256
`C269F9D7E9DEDC23118514FED8EBC0987500948ABF88F8A2129EA0F264315A34`.

The timestamped approximately 30 s FC/BMS/EMS exports are CSV. They are useful
evidence that device measurements exist, but are not authorized formal raw facts
for v2. `Cleaned_Power_Data.xlsx` is a processed aggregate whose `Time` data
cells are blank; rename helpers and generated/digitized workbooks likewise do
not satisfy the raw-measurement gate. See `docs/v2_raw_excel_inventory.md`.

## Access and preflight policy

Method selection and calibration are Train-only. `Train` is matched
case-insensitively. Validation and Test are rejected before a payload loader can
run. The provenance preflight separately reports absent usable raw measurements
and absent technical-specification evidence; it does not collapse those causes
into a generic `BLOCKED` result. Technical-specification evidence is represented
by a validated record rather than a caller-supplied availability flag. The
record requires an authoritative PDF classification, a 64-hex-digit SHA-256,
a positive page count, and explicit source identifier and reference fields.

Current formal dataset status: **UNCALIBRATED / NO-GO**. Resolution requires
source-owner clarification or an authorized, traceable conversion of the
timestamped device exports. Any later shore-power channel must be identified as
measured; terminal charging inferred from SOC remains modeled rather than
measured.

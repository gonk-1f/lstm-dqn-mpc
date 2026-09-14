# v2 Raw Excel Inventory

Audit date: 2026-09-14.

Authoritative user-data root (files were inspected in place and were not copied
into this repository):
`C:\Users\20883\OneDrive\Desktop\氢舟一号`.

## Inventory result

A forced recursive enumeration found 71 `.xlsx` paths: 69 ordinary files and
2 temporary lock files. There are no `.xls` files. An Excel suffix makes a file
an accepted **candidate** only; it does not establish that the workbook contains
usable original telemetry.

| Workbook or group | Sheet | Column | Unit | Timestamp | Actual sampling interval | Missing rate | Physical meaning | Usable | Reason |
|---|---|---|---|---|---|---|---|---|---|
| `1修改名称.xlsx` (57 files) | helper sheet | formula cells constructing `ren ...` commands | not applicable | none | not applicable | not applicable | filename-renaming command helper | no | Administrative helper, not measurement telemetry |
| `~$1修改名称.xlsx` (2 files) | not audited | not audited | not audited | not audited | not audited | not audited | temporary Excel lock file | no | Temporary lock artifact |
| `Cleaned_Power_Data.xlsx` | `Sheet1` | `Time` | unresolved | column exists, all data cells blank | cannot be established | `Time`: 100% for 51,487 data rows | intended time index for processed aggregate | no | Processed aggregate with no usable timestamps |
| `Cleaned_Power_Data.xlsx` | `Sheet1` | `Total_System_kW` | kW (from name) | none usable | cannot be established | not promoted | processed total-system power | no | Filename/content indicate a cleaned aggregate, not an original device export |
| `Cleaned_Power_Data.xlsx` | `Sheet1` | `FC_Stack_kW` | kW (from name) | none usable | cannot be established | not promoted | processed fuel-cell stack power | no | Same processed aggregate and missing timestamp basis |
| `Cleaned_Power_Data.xlsx` | `Sheet1` | `Battery_kW` | kW (from name) | none usable | cannot be established | not promoted | processed battery power | no | Same processed aggregate and missing timestamp basis |
| `ship_*.xlsx` | varies | generated operating-condition/load fields | varies | generated/model index | generated/resampled | not promoted | synthetic or derived ship operating profiles | no | Generated derivative, not an original measurement source |
| `extracted_curves*.xlsx`, `cv2.xlsx` | varies | digitized curve points | varies | none | not applicable | not promoted | image/digitization derivative | no | Image transcription cannot be promoted to raw fact |
| `燃料电池总功率汇总结果.xlsx`, `TotalPower_Result.xlsx` | varies | aggregated power fields | varies | no authorized original timestamp basis established | unresolved | not promoted | calculated/aggregated power result | no | Processed result, not an authorized original device export |

`Cleaned_Power_Data.xlsx` has worksheet dimension `A1:D51488` (51,488
worksheet rows including the header). Direct OOXML inspection found only the
header cell populated in column A, so all 51,487 `Time` data cells are blank.
Its SHA-256 is
`CC42029B1A62714B923630E239C4D5529B788DA75740050BDC8C282D2C8749D4`.
The inspection used Python's standard-library ZIP/XML readers and introduces no
runtime dependency on `openpyxl`.

## Gate decision

Timestamped FC/BMS/EMS device exports at approximately 30 s intervals do exist
under the authoritative root as CSV files. The current source contract does not
authorize historical CSV as a formal v2 raw fact. The repository's MAT files,
1 s interpolation products, generated workbooks, and image transcriptions are
also derivatives and remain excluded.

Consequently, `raw_measurements_available` is false for the audited inventory.
Formal `mode_aware_operating_cycle_v2` rebuild status is **NO-GO**, pending
clarification of the original source boundary or an explicitly authorized,
traceable raw conversion.

# Repository cleanup and verification evidence (2026-10-08)

Base: `d111c99`. No official training or Test trajectory access.

- [Cleanup and dependency report](../../v4_repository_audit_2026-10-08.md)
- [TD scale diagnosis](../../v4_td_numerical_diagnostic_2026-10-08.md)
- [Before inventory and protected SHA256](before_inventory.json)
- [Source, tests, calibration and dependency audit](dependency_audit.json)
- [Moved documents with old/new hashes](documentation_moves.json)
- [Page snapshot origins](page_snapshots.json)
- [Executed deletion file list](approved_deletions.json)
- [Deferred cache and temporary deletion list](final_temporary_cleanup.json)
- [Baseline documentation audit](docs_audit.json)
- [Final documentation/link review](final_docs_review.json)
- [Final numerical implementation review](final_numerical_review.json)
- [Main regression log](regression.log)
- [Actual nonzero SHORE contracts](actual_shore_contract.log)
- [Final CLI and SHORE contracts](final_cli_shore_contract.log)
- [Final counts, hashes and verification](final_verification.json)

The main regression had 182 passed and 330 subtests passed. The subsequent final CLI/actual-SHORE run had 10 passed, with two cases already covered: 190 distinct test cases overall. The SHA256 records describe working-tree bytes, which may include Windows CRLF; comparison to Git uses canonical LF where relevant. Metadata manifests are hashed without reading any trajectory payload.

The deferred cleanup was rejected by automatic approval (`blocked by policy`); its files remain. The executed 344-file cleanup list is separate. All existing evidence archives remain byte-identical.

# Uniform economic replay scaling: implementation verification

Base `3d75e5a`. This archive contains synthetic/unit verification only. The user will launch the single formal40-round run; no official Train/Validation/Test trajectory was read here.

- [Implementation and PyCharm command](../../v4_reward_scale_training_2026-10-08.md)
- [Verification, invariant hashes and synthetic update counts](verification.json)
- [Final regression:143 passed](regression_final.log)
- [Core RED](core_red.log) / [GREEN](core_green.log)
- [Windows long-path and FC roundoff RED](winpath_roundoff_red.log) / [GREEN20](scaled_diagnostics_final.log)
- [Previously failing CLI case after Windows fix](entry_windows_image_final.log)
- [Pre-change protected hashes](before_protected_sha256.json)

The first combined run had135pass/1failure because a260-character image filename failed in Pillow on Windows. After extended-path handling and seven added regression cases, the full suite passed143tests. Existing archives and physical/economic sources were retained byte-identically.

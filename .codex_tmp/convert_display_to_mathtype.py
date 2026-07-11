from __future__ import annotations

import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path


def setup_pywin32() -> None:
    base = Path(".codex_tmp/pydeps").resolve()
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(base / "win32"))
    sys.path.insert(0, str(base / "win32/lib"))
    os.add_dll_directory(str(base / "pywin32_system32"))


def parse_display_formulas(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    marker = "一、单独成行公式 MathType/LaTeX 代码"
    next_marker = "二、正文内行内公式/变量替换清单"
    start = text.index(marker)
    end = text.index(next_marker)
    part = text[start:end]
    formulas: list[tuple[str, str]] = []
    for m in re.finditer(r"（([23]-\d+)）\s*\n(.*?)(?=\n（[23]-\d+）|\n={10,}|\Z)", part, re.S):
        key = m.group(1)
        latex = m.group(2).strip()
        if latex:
            formulas.append((key, normalize_for_mathtype(latex)))
    return formulas


def normalize_for_mathtype(latex: str) -> str:
    s = latex.strip()
    s = re.sub(r"\\operatorname\{([^{}]+)\}", r"\\mathrm{\1}", s)
    # MathType's TeX Toggle recognizes dollar-delimited TeX most reliably.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s*\n\s*", " ", s)
    return s


def inspect_docx(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        embeds = [n for n in names if n.startswith("word/embeddings/")]
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    return {
        "embeddings": len(embeds),
        "ole": xml.count("o:OLEObject"),
        "omml": xml.count("<m:oMath"),
        "progids": re.findall(r'ProgID="([^"]+)"', xml),
    }


def main() -> int:
    setup_pywin32()
    import pythoncom
    import win32com.client

    source = Path(os.environ["MATHTYPE_SOURCE_DOCX"]).resolve()
    formula_list = Path(os.environ["MATHTYPE_FORMULA_LIST"]).resolve()
    output = Path(os.environ["MATHTYPE_OUTPUT_DOCX"]).resolve()
    max_count = int(os.environ.get("MATHTYPE_MAX_COUNT", "0"))

    formulas = parse_display_formulas(formula_list)
    if max_count:
        formulas = formulas[:max_count]

    shutil.copy2(source, output)
    print(f"formulas={len(formulas)} output={output}")

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0

    converted: list[str] = []
    failed: list[tuple[str, str]] = []
    try:
        doc = word.Documents.Open(str(output))
        para_start = 1
        for key, latex in formulas:
            target = f"\uff08{key}\uff09"
            found_idx = None
            for i in range(para_start, doc.Paragraphs.Count + 1):
                para_i = doc.Paragraphs(i)
                ptxt = para_i.Range.Text
                try:
                    has_math = para_i.Range.OMaths.Count > 0
                except Exception:
                    has_math = False
                try:
                    in_table = para_i.Range.Information(12)  # wdWithInTable
                except Exception:
                    in_table = False
                # Display-formula paragraphs contain a normal-text equation
                # number plus an OMML object. Captions/tables may contain the
                # same number text but do not have OMath objects.
                if target in ptxt and has_math and not in_table:
                    found_idx = i
                    break
            if found_idx is None:
                failed.append((key, "formula number paragraph not found"))
                continue

            scratch = None
            try:
                scratch = word.Documents.Add()
                word.Selection.TypeText(f"${latex}$")
                scratch.Range(scratch.Content.Start, scratch.Content.Start + len(latex) + 2).Select()
                word.Application.Run("MTCommand_TeXToggle")
                if scratch.InlineShapes.Count < 1:
                    failed.append((key, "TeX Toggle did not create MathType OLE in scratch document"))
                    scratch.Close(False)
                    continue
                scratch.InlineShapes(1).Range.Copy()
            except Exception as exc:
                failed.append((key, f"scratch MathType conversion failed: {exc}"))
                if scratch is not None:
                    try:
                        scratch.Close(False)
                    except Exception:
                        pass
                continue

            para = doc.Paragraphs(found_idx)
            rng = para.Range
            # Replace only the content before the paragraph mark. Word rejects
            # assigning paragraph marks into ranges that currently contain OMML.
            content_rng = doc.Range(rng.Start, rng.End - 1)
            content_rng.Select()
            word.Selection.Delete()
            before = doc.InlineShapes.Count
            word.Selection.Paste()
            word.Selection.TypeText(f"\t{target}")
            para = doc.Paragraphs(found_idx)
            para.Range.ParagraphFormat.Alignment = 1  # wdAlignParagraphCenter
            after = doc.InlineShapes.Count
            try:
                scratch.Close(False)
            except Exception:
                pass
            if after <= before:
                failed.append((key, "pasting MathType OLE did not add an InlineShape"))
            else:
                converted.append(key)
            para_start = found_idx + 1

        doc.Save()
        doc.Close(False)
    finally:
        try:
            word.Quit()
        except Exception:
            pass

    info = inspect_docx(output)
    print(f"converted={len(converted)} failed={len(failed)}")
    print(f"embeddings={info['embeddings']} ole={info['ole']} omml={info['omml']}")
    for key, reason in failed:
        print(f"FAILED\t{key}\t{reason}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

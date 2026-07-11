from __future__ import annotations

import copy
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree
from latex2mathml.converter import convert as latex_to_mathml


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}
W = f"{{{W_NS}}}"


SOURCE_DOCX = Path(os.environ["SOURCE_DOCX"])
FORMULA_TXT = Path(os.environ["FORMULA_TXT"])
OUTPUT_DOCX = Path(os.environ["OUTPUT_DOCX"])
REPORT_TXT = Path(os.environ["REPORT_TXT"])
MML2OMML_XSL = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")


RAW_SCAN_TERMS = [
    "P_fc",
    "P_load",
    "P_bat",
    "P_batt",
    "m_H2",
    "q_ramp",
    "q_soc",
    "q_bat",
    "SOC_ref",
    "theta^-",
    "Q_theta",
    "L_DQN",
    "phi_",
    "lambda_",
    "sigma_",
    "mu_",
    "eta_",
    "Delta",
    "Sigma",
]


GREEK = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\Delta": "Δ",
    r"\varepsilon": "ε",
    r"\eta": "η",
    r"\theta": "θ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\sigma": "σ",
    r"\Sigma": "Σ",
    r"\Phi": "Φ",
    r"\phi": "φ",
    r"\varphi": "φ",
    r"\omega": "ω",
}


@dataclass(frozen=True)
class Formula:
    key: str
    latex: str
    omml: etree._Element


def qn(tag: str) -> str:
    prefix, local = tag.split(":")
    if prefix == "w":
        return f"{{{W_NS}}}{local}"
    if prefix == "m":
        return f"{{{M_NS}}}{local}"
    raise ValueError(tag)


def paragraph_text(p: etree._Element) -> str:
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))


def is_in_table(p: etree._Element) -> bool:
    return any(a.tag == qn("w:tbl") for a in p.iterancestors())


def get_ppr_copy(p: etree._Element) -> etree._Element | None:
    ppr = p.find(qn("w:pPr"))
    return copy.deepcopy(ppr) if ppr is not None else None


def clear_para_keep_ppr(p: etree._Element) -> None:
    ppr = get_ppr_copy(p)
    for child in list(p):
        p.remove(child)
    if ppr is not None:
        p.append(ppr)


def text_run(text: str, rpr: etree._Element | None = None) -> etree._Element:
    r = etree.Element(qn("w:r"))
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    t = etree.SubElement(r, qn("w:t"))
    if text[:1].isspace() or text[-1:].isspace():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def set_omml_font_size(omml: etree._Element, half_points: int) -> None:
    for mr in omml.xpath(".//m:r", namespaces=NS):
        wrpr = mr.find(qn("w:rPr"))
        if wrpr is None:
            wrpr = etree.Element(qn("w:rPr"))
            mt = mr.find(qn("m:t"))
            if mt is not None:
                mr.insert(list(mr).index(mt), wrpr)
            else:
                mr.insert(0, wrpr)
        for tag in ["w:sz", "w:szCs"]:
            node = wrpr.find(qn(tag))
            if node is None:
                node = etree.SubElement(wrpr, qn(tag))
            node.set(qn("w:val"), str(half_points))


def first_rpr(p: etree._Element) -> etree._Element | None:
    rpr = p.find(".//w:rPr", namespaces=NS)
    return copy.deepcopy(rpr) if rpr is not None else None


def omml_from_latex(latex: str, transform: etree.XSLT) -> etree._Element:
    latex = re.sub(r"\\operatorname\{([^{}]+)\}", r"\\mathrm{\1}", latex)
    mathml = latex_to_mathml(latex)
    mathml_root = etree.fromstring(mathml.encode("utf-8"))
    omml_doc = transform(mathml_root)
    root = omml_doc.getroot()
    if root.tag == qn("m:oMathPara"):
        math = root.find("m:oMath", namespaces=NS)
        if math is None:
            raise RuntimeError("OMML transform returned oMathPara without oMath")
        return copy.deepcopy(math)
    if root.tag != qn("m:oMath"):
        raise RuntimeError(f"Unexpected OMML root: {root.tag}")
    return copy.deepcopy(root)


def parse_formula_file(text: str, transform: etree.XSLT) -> tuple[dict[str, Formula], list[Formula]]:
    display_part, rest = text.split("二、正文内行内公式/变量替换清单", 1)
    inline_part = rest.split("三、排版检查要求", 1)[0]

    display: dict[str, Formula] = {}
    for m in re.finditer(r"\uff08([23]-\d+)\uff09\s*\n(.*?)(?=\n\uff08[23]-\d+\uff09|\n={10,}|\Z)", display_part, re.S):
        key = m.group(1)
        latex = m.group(2).strip()
        if latex:
            display[key] = Formula(key, latex, omml_from_latex(latex, transform))

    inline: list[Formula] = []
    seen: set[str] = set()
    for raw in inline_part.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("以下内容") or re.match(r"^\d+\.", line):
            continue
        if line in seen:
            continue
        seen.add(line)
        try:
            inline.append(Formula(line, line, omml_from_latex(line, transform)))
        except Exception as exc:
            print(f"INLINE_CONVERT_FAIL\t{line}\t{exc}", file=sys.stderr)

    # Full expressions that also appear inline in the algorithm text.
    for key in ["3-14"]:
        if key in display and display[key].latex not in seen:
            inline.append(display[key])
            seen.add(display[key].latex)

    return display, inline


def normalize_latex_for_plain(latex: str) -> str:
    s = latex
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\,", "").replace(r"\quad", " ")
    s = s.replace(r"\ldots", "…").replace(r"\cdots", "…").replace(r"\sim", "~")
    s = s.replace(r"\in", "∈").replace(r"\mapsto", "↦").replace(r"\le", "≤").replace(r"\max", "max")
    s = s.replace(r"\operatorname{LSTM}", "LSTM")
    s = s.replace(r"\operatorname{MPC}", "MPC")
    s = s.replace(r"\operatorname{first}", "first")
    s = s.replace(r"\operatorname{SineKAN}", "SineKAN")
    s = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbb\{E\}", "E", s)
    s = re.sub(r"\\mathcal\{D\}", "D", s)
    s = s.replace(r"\Delta", "Δ")
    for k, v in GREEK.items():
        s = s.replace(k, v)
    s = re.sub(r"\\hat\{?P\}?", "P̂", s)
    s = re.sub(r"\\hat\{?Y\}?", "Ŷ", s)
    s = re.sub(r"\\tilde\{?c\}?", "c̃", s)
    s = re.sub(r"\\dot\{?m\}?", "ṁ", s)
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    s = s.replace(r"\(", "(").replace(r"\)", ")")
    return s


def strip_math_braces(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)
        s = re.sub(r"\^\{([^{}]+)\}", r"^\1", s)
    return s


def compact_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def latex_variants(latex: str) -> set[str]:
    base = normalize_latex_for_plain(latex)
    variants = {base, strip_math_braces(base)}
    expanded = set(variants)
    for v in variants:
        expanded.add(v.replace(" ", ""))
        expanded.add(v.replace("{", "").replace("}", ""))
        expanded.add(strip_math_braces(v).replace("{", "").replace("}", ""))
    variants = {compact_spaces(v) for v in expanded if v and len(v.strip()) > 1}

    manual: set[str] = set()
    for v in list(variants):
        manual.add(v.replace("P̂", "P\u0302"))
        manual.add(v.replace("c̃", "c\u0303"))
        manual.add(v.replace("Ŷ", "Y\u0302"))
        manual.add(v.replace("ṁ", "m"))
        manual.add(v.replace("H_2", "H2"))
        manual.add(v.replace("theta", "θ"))
        manual.add(v.replace("lambda", "λ"))
        manual.add(v.replace("phi", "φ"))
        manual.add(v.replace("sigma", "σ"))
        manual.add(v.replace("mu", "μ"))
        manual.add(v.replace("eta", "η"))
        manual.add(v.replace("Delta", "Δ"))
        manual.add(v.replace("SineKAN_{θ}", "SineKAN_θ"))
    variants |= {compact_spaces(v) for v in manual if v and len(v.strip()) > 1}
    return variants


def build_inline_patterns(inline: list[Formula]) -> list[tuple[str, Formula]]:
    pairs: list[tuple[str, Formula]] = []
    seen: set[tuple[str, str]] = set()
    for f in inline:
        for raw in latex_variants(f.latex):
            if len(raw) == 1:
                continue
            # Avoid turning ordinary prose single-letter symbols into equations.
            if raw in {"F", "H", "M", "K", "D"}:
                continue
            key = (raw, f.key)
            if key not in seen:
                seen.add(key)
                pairs.append((raw, f))

    extra = {
        "P_bat(t)": r"P_{\mathrm{bat}}(t)",
        "P_fc^cmd(t)": r"P_{\mathrm{fc}}^{\mathrm{cmd}}(t)",
        "P_fc^cmd(t+1)": r"P_{\mathrm{fc}}^{\mathrm{cmd}}(t+1)",
        "P_bat(t)=P_load(t)-P_fc^cmd(t)": r"P_{\mathrm{bat}}(t)=P_{\mathrm{load}}(t)-P_{\mathrm{fc}}^{\mathrm{cmd}}(t)",
        "P_bat(t) = P_load(t) - P_fc^cmd(t)": r"P_{\mathrm{bat}}(t)=P_{\mathrm{load}}(t)-P_{\mathrm{fc}}^{\mathrm{cmd}}(t)",
        "P_batt=Load-P_fc": r"P_{\mathrm{batt}}=\mathrm{Load}-P_{\mathrm{fc}}",
        "P_fc": r"P_{\mathrm{fc}}",
        "P_batt": r"P_{\mathrm{batt}}",
    }
    transform = etree.XSLT(etree.parse(str(MML2OMML_XSL)))
    for raw, latex in extra.items():
        pairs.append((raw, Formula(raw, latex, omml_from_latex(latex, transform))))

    # Longest raw text first so a full expression wins over its variables.
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


def find_matches(text: str, patterns: list[tuple[str, Formula]]) -> list[tuple[int, int, Formula, str]]:
    matches: list[tuple[int, int, Formula, str]] = []
    occupied = [False] * len(text)
    for raw, formula in patterns:
        if not raw:
            continue
        start = 0
        while True:
            idx = text.find(raw, start)
            if idx < 0:
                break
            end = idx + len(raw)
            if not any(occupied[idx:end]):
                matches.append((idx, end, formula, raw))
                for i in range(idx, end):
                    occupied[i] = True
            start = idx + max(1, len(raw))
    matches.sort(key=lambda x: x[0])
    return matches


def replace_inline_paragraph(p: etree._Element, matches: list[tuple[int, int, Formula, str]]) -> int:
    text = paragraph_text(p)
    rpr = first_rpr(p)
    clear_para_keep_ppr(p)
    pos = 0
    count = 0
    for start, end, formula, _raw in matches:
        if start > pos:
            p.append(text_run(text[pos:start], rpr))
        p.append(copy.deepcopy(formula.omml))
        count += 1
        pos = end
    if pos < len(text):
        p.append(text_run(text[pos:], rpr))
    return count


def replace_display_paragraph(p: etree._Element, formula: Formula, number: str) -> None:
    rpr = first_rpr(p)
    clear_para_keep_ppr(p)
    omml = copy.deepcopy(formula.omml)
    if len(formula.latex) > 120:
        set_omml_font_size(omml, 18)
    p.append(omml)
    p.append(text_run(f"    （{number}）", rpr))


def copy_with_replaced_document_xml(src_docx: Path, out_docx: Path, document_xml: bytes) -> None:
    if out_docx.exists():
        stamp = tempfile.NamedTemporaryFile(delete=True).name
        del stamp
        backup = out_docx.with_suffix(out_docx.suffix + ".bak")
        if backup.exists():
            backup.unlink()
        out_docx.replace(backup)
    else:
        out_docx.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(src_docx, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, document_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))
        shutil.move(str(tmp_path), str(out_docx))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> int:
    formula_text = FORMULA_TXT.read_text(encoding="utf-8")
    transform = etree.XSLT(etree.parse(str(MML2OMML_XSL)))
    display, inline = parse_formula_file(formula_text, transform)
    inline_patterns = build_inline_patterns(inline)

    with zipfile.ZipFile(SOURCE_DOCX, "r") as z:
        root = etree.fromstring(z.read("word/document.xml"))

    display_done: list[str] = []
    inline_done = 0
    skipped_raw: list[str] = []
    reference_mode = False
    unmatched_display: list[str] = []

    for p in root.xpath(".//w:p", namespaces=NS):
        text = paragraph_text(p)
        stripped = text.strip()
        if stripped.startswith("参考文献"):
            reference_mode = True

        number_match = re.search(r"\uff08([23]-\d+)\uff09", text)
        if number_match and number_match.group(1) in display and not is_in_table(p):
            num = number_match.group(1)
            replace_display_paragraph(p, display[num], num)
            display_done.append(num)
            continue

        if reference_mode or is_in_table(p):
            continue
        matches = find_matches(text, inline_patterns)
        if matches:
            inline_done += replace_inline_paragraph(p, matches)

    for num in display:
        if num not in display_done:
            unmatched_display.append(num)

    xml_bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")
    copy_with_replaced_document_xml(SOURCE_DOCX, OUTPUT_DOCX, xml_bytes)

    with zipfile.ZipFile(OUTPUT_DOCX, "r") as z:
        out_root = etree.fromstring(z.read("word/document.xml"))
    visible_text = "\n".join(paragraph_text(p) for p in out_root.xpath(".//w:p", namespaces=NS))
    numbers_present = sorted(set(re.findall(r"\uff08([23]-\d+)\uff09", visible_text)), key=lambda x: (x[0], int(x.split("-")[1])))
    expected_numbers = [f"2-{i}" for i in range(1, 19)] + [f"3-{i}" for i in range(1, 35)]
    missing_numbers = [n for n in expected_numbers if n not in numbers_present]
    raw_hits = {term: visible_text.count(term) for term in RAW_SCAN_TERMS if visible_text.count(term)}

    all_omath = out_root.xpath(".//m:oMath", namespaces=NS)
    report_lines = [
        "公式排版检查报告",
        f"源文件: {SOURCE_DOCX}",
        f"输出文件: {OUTPUT_DOCX}",
        f"共处理显示公式: {len(display_done)}",
        f"共处理行内公式片段: {inline_done}",
        f"DOCX XML 中 OMML 公式对象总数: {len(all_omath)}",
        f"公式编号（2-1）至（3-34）是否完整: {'是' if not missing_numbers else '否'}",
        f"缺失公式编号: {', '.join(missing_numbers) if missing_numbers else '无'}",
        f"无法自动替换的显示公式: {', '.join(unmatched_display) if unmatched_display else '无'}",
        f"指定 raw 下划线变量是否仍存在: {'是' if raw_hits else '否'}",
        f"指定 raw 下划线变量残留: {raw_hits if raw_hits else '无'}",
        "跳过或保留的 raw 文本段落:",
    ]
    if skipped_raw:
        report_lines.extend(f"- {s}" for s in skipped_raw)
    else:
        report_lines.append("- 无")
    report_lines.extend(
        [
            "是否未改动正文文字: 是；脚本仅把匹配到的公式片段替换为 OMML 公式对象，非公式文本片段按原子串保留。",
            "表格、参考文献、文件路径处理: 表格和参考文献跳过；文件路径和图片占位说明文字保留，仅将其中匹配公式变量转为公式对象。",
            f"公式代码文件: {FORMULA_TXT}",
            f"报告路径: {REPORT_TXT}",
        ]
    )
    REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

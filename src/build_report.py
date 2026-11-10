"""src/build_report.py
Generates the formal academic technical report:
  Report/UCS321_AQI_Forecasting_Report.docx
and converts it to PDF via MS Word COM (updating TOC page fields) or docx2pdf.
"""

import os
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

os.makedirs('Report', exist_ok=True)
DOCX_PATH = os.path.abspath('Report/UCS321_AQI_Forecasting_Report.docx')
PDF_PATH = os.path.abspath('Report/UCS321_AQI_Forecasting_Report.pdf')


# ---------------------------------------------------------------------------
# XML Helpers for Styling, TOC, Page Numbers, Table Borders & Shading
# ---------------------------------------------------------------------------
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=110, right=110):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_table_borders(table, border_color="CCCCCC"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="8" w:space="0" w:color="1B365D"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="1B365D"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def add_page_number_field(run):
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls('w'))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


def add_toc_field(doc):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run()
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> TOC \o "1-1" \h \z \u </w:instrText>' % nsdecls('w'))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run_hint = p.add_run("Right-click and select 'Update Field' in Microsoft Word if page numbers are not visible.")
    run_hint.font.name = 'Times New Roman'
    run_hint.font.size = Pt(11)
    run_hint.italic = True
    run_end = p.add_run()
    run_end._r.append(fldChar3)


def configure_document(doc):
    section = doc.sections[0]
    # A4 dimensions
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.different_first_page_header_footer = True

    # Base Normal style
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Heading 1 (14 pt bold)
    h1 = doc.styles['Heading 1']
    h1.font.name = 'Times New Roman'
    h1.font.size = Pt(14)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)
    h1.paragraph_format.line_spacing = 1.15
    h1.paragraph_format.keep_with_next = True

    # Heading 2 (13 pt bold)
    h2 = doc.styles['Heading 2']
    h2.font.name = 'Times New Roman'
    h2.font.size = Pt(13)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(4)
    h2.paragraph_format.line_spacing = 1.15
    h2.paragraph_format.keep_with_next = True

    # Heading 3 (12 pt bold)
    h3 = doc.styles['Heading 3']
    h3.font.name = 'Times New Roman'
    h3.font.size = Pt(12)
    h3.font.bold = True
    h3.font.color.rgb = RGBColor(0x2B, 0x4C, 0x7E)
    h3.paragraph_format.space_before = Pt(8)
    h3.paragraph_format.space_after = Pt(3)
    h3.paragraph_format.line_spacing = 1.15
    h3.paragraph_format.keep_with_next = True

    # Footer with page numbers
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_left = fp.add_run("UCS321 – EST Project Report  |  Page ")
    r_left.font.name = 'Times New Roman'
    r_left.font.size = Pt(10)
    r_left.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    r_pg = fp.add_run()
    r_pg.font.name = 'Times New Roman'
    r_pg.font.size = Pt(10)
    r_pg.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    add_page_number_field(r_pg)


# Track figures and tables for List of Figures / List of Tables
FIGURE_LIST = []
TABLE_LIST = []


def add_body(doc, text, bold_prefix=None, space_after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(space_after)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = 'Times New Roman'
        rb.font.size = Pt(12)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    return p


def add_bullet(doc, bold_prefix, text):
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(4)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = 'Times New Roman'
        rb.font.size = Pt(12)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    return p


def add_quote(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.right_indent = Inches(0.4)
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.italic = True
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11.5)
    r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)


def add_figure(doc, img_rel_path, fig_num, caption, width_in=6.15):
    full_caption = f"Figure {fig_num}: {caption}"
    FIGURE_LIST.append(full_caption)
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(3)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    run = p_img.add_run()
    run.add_picture(img_rel_path, width=Inches(6.0))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.line_spacing = 1.1
    p_cap.paragraph_format.space_after = Pt(6)
    rc = p_cap.add_run(full_caption)
    rc.font.name = 'Times New Roman'
    rc.font.size = Pt(10.5)
    rc.italic = True
    rc.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def add_styled_table(doc, table_num, caption, headers, rows, col_widths=None, highlight_rows=None):
    full_caption = f"Table {table_num}: {caption}"
    TABLE_LIST.append(full_caption)
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(4)
    p_cap.paragraph_format.line_spacing = 1.15
    p_cap.paragraph_format.keep_with_next = True
    rc = p_cap.add_run(full_caption)
    rc.font.name = 'Times New Roman'
    rc.font.size = Pt(10.5)
    rc.bold = True
    rc.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)

    # Header row
    hdr_cells = table.rows[0].cells
    # Repeat header across pages
    trPr = table.rows[0]._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:tblHeader'))
    for j, h_text in enumerate(headers):
        cell = hdr_cells[j]
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.1
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(h_text))
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        if col_widths:
            cell.width = Inches(col_widths[j])

    # Data rows
    highlight_rows = highlight_rows or set()
    for i, row_data in enumerate(rows):
        row = table.rows[i + 1]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        is_hl = i in highlight_rows
        bg = "E8F0FE" if is_hl else ("F8FAFC" if i % 2 == 1 else "FFFFFF")
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=50, bottom=50, left=75, right=75)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.08
            p.paragraph_format.space_after = Pt(0)
            if len(rows) <= 12 and i < len(rows) - 1:
                p.paragraph_format.keep_with_next = True
            r = p.add_run(str(val))
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            if is_hl:
                r.bold = True
            if col_widths:
                cell.width = Inches(col_widths[j])

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(4)
    spacer.paragraph_format.line_spacing = 1.0


def build_report():
    doc = Document()
    configure_document(doc)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    for _ in range(3):
        sp = doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(6)

    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.line_spacing = 1.3
    r_inst = p_inst.add_run("THAPAR INSTITUTE OF ENGINEERING & TECHNOLOGY, PATIALA")
    r_inst.font.name = 'Times New Roman'
    r_inst.font.size = Pt(14)
    r_inst.bold = True
    r_inst.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_course = doc.add_paragraph()
    p_course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_course.paragraph_format.space_after = Pt(28)
    r_course = p_course.add_run("UCS321 – EST Project")
    r_course.font.name = 'Times New Roman'
    r_course.font.size = Pt(13)
    r_course.bold = True
    r_course.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.line_spacing = 1.35
    p_title.paragraph_format.space_after = Pt(16)
    r_title = p_title.add_run("Forecasting Air Quality Index around IndianOil Refinery Zones using ARIMA and LSTM Models")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(20)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(22)
    r_sub = p_sub.add_run("Technical Project Report — Chennai Corridor (CPCL Manali Refinery Zone)")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(12.5)
    r_sub.italic = True
    r_sub.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    p_grp = doc.add_paragraph()
    p_grp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_grp.paragraph_format.space_after = Pt(8)
    r_grp = p_grp.add_run("Submitted by: Group 2W1")
    r_grp.font.name = 'Times New Roman'
    r_grp.font.size = Pt(12.5)
    r_grp.bold = True
    r_grp.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    members = [
        ("Arush Saini", "1025230031"),
        ("Jiya Wadhwa", "1025230094"),
        ("Darsh Marwaha", "1025230032"),
        ("Tahoor", "1025230126"),
        ("Arshita Doomra", "1025230123"),
    ]
    m_tbl = doc.add_table(rows=1 + len(members), cols=2)
    m_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    m_tbl.autofit = False
    for r_idx, row_vals in enumerate([("Name", "Roll No.")] + members):
        row = m_tbl.rows[r_idx]
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = Inches(2.5 if c_idx == 0 else 1.6)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            if r_idx == 0:
                tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="1B365D"/>'))
            elif r_idx % 2 == 0:
                tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="F1F5F9"/>'))
            tcPr.append(parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
                f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
                f'<w:left w:val="none"/><w:right w:val="none"/>'
                f'</w:tcBorders>'
            ))
            tcPr.append(parse_xml(
                f'<w:tcMar {nsdecls("w")}>'
                f'<w:top w:w="70" w:type="dxa"/><w:bottom w:w="70" w:type="dxa"/>'
                f'<w:left w:w="120" w:type="dxa"/><w:right w:w="120" w:type="dxa"/>'
                f'</w:tcMar>'
            ))
            p_c = cell.paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p_c.paragraph_format.space_before = Pt(0)
            p_c.paragraph_format.space_after = Pt(0)
            rc = p_c.add_run(val)
            rc.font.name = 'Times New Roman'
            rc.font.size = Pt(11)
            if r_idx == 0:
                rc.bold = True
                rc.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                rc.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.line_spacing = 1.5
    p_meta.paragraph_format.space_before = Pt(22)
    p_meta.paragraph_format.space_after = Pt(12)
    for lbl, val in [
        ("Course: ", "UCS321 – EST Project\n"),
        ("Institute: ", "Thapar Institute of Engineering & Technology, Patiala\n"),
        ("Date: ", "November 2026")
    ]:
        rl = p_meta.add_run(lbl)
        rl.bold = True
        rl.font.name = 'Times New Roman'
        rl.font.size = Pt(11.5)
        rv = p_meta.add_run(val)
        rv.font.name = 'Times New Roman'
        rv.font.size = Pt(11.5)

    doc.add_page_break()

    # =========================================================================
    # FRONT MATTER: ABSTRACT, TOC, LIST OF FIGURES, LIST OF TABLES
    # =========================================================================
    doc.add_heading("Abstract", level=1)
    add_body(
        doc,
        "This project evaluates classical state-space and deep learning sequence models for forecasting daily "
        "Air Quality Index (AQI) around the Chennai Petroleum Corporation Limited (CPCL) Manali refinery corridor, "
        "a subsidiary of Indian Oil Corporation Limited (IOCL). Using daily Central Pollution Control Board (CPCB) "
        "ambient monitoring data for Chennai from 2016-01-01 to 2020-06-30 (1,643 calendar days, excluding a 2015 "
        "monitoring outage), the dataset is partitioned chronologically into Train (1,096 days), Validation (243 days), "
        "Test (182 days), and an untouched COVID-19 lockdown Stress window (122 days). Strict leakage controls govern "
        "15 anomaly-corrected days, gap-adaptive imputation, upper-bound winsorization, and train-only scaling. "
        "Across 1-day-ahead forecasting on the Test split, ARIMAX(2,1,1) with three annual Fourier harmonics and lagged "
        "PM2.5 achieved the lowest error (RMSE 25.97, MAE 18.45, R² 0.6657, CPCB category accuracy 76.92%), outperforming "
        "a hybrid ARIMAX-LSTM residual model (RMSE 26.81, MAE 19.91), a tuned CNN-LSTM (RMSE 27.63, MAE 21.21), Pure ARIMA "
        "(RMSE 30.58, MAE 19.23), Univariate LSTM (RMSE 31.73, MAE 22.87), and naive persistence (RMSE 32.10, MAE 19.56). "
        "In 7-day-ahead evaluation across 175 origins, a validation-selected Epoch-2 Direct Multi-Step LSTM beats persistence "
        "only at horizons 5–7, while recursive ARIMA-Fourier leads horizons 2–6. During the lockdown stress window (mean AQI 78.81), "
        "Pure ARIMA achieved the lowest RMSE (13.51) followed by persistence (14.45), whereas exogenous and deep models degraded. "
        "Split-conformal intervals attained 96.70% empirical test coverage."
    )

    doc.add_page_break()
    doc.add_heading("Table of Contents", level=1)
    add_toc_field(doc)

    # Note: We will populate List of Figures and List of Tables after building sections using placeholder paragraphs
    doc.add_page_break()
    doc.add_heading("List of Figures", level=1)
    p_lof_placeholder = doc.add_paragraph()

    doc.add_heading("List of Tables", level=1)
    p_lot_placeholder = doc.add_paragraph()

    doc.add_page_break()

    # =========================================================================
    # SECTION 1: INTRODUCTION
    # =========================================================================
    doc.add_heading("1. Introduction", level=1)
    doc.add_heading("1.1 Air Quality Monitoring in Industrial Refinery Zones", level=2)
    add_body(
        doc,
        "Urban and industrial air quality management relies on timely quantification of criteria pollutants regulated "
        "under national ambient standards. Petroleum refining complexes handle high-temperature catalytic cracking, sulfur "
        "recovery, combustion in process heaters and boilers, and hydrocarbon storage, which collectively emit particulate "
        "matter (PM2.5, PM10), nitrogen oxides (NOx), sulfur dioxide (SO2), carbon monoxide (CO), and volatile organic "
        "compounds (VOCs). In coastal industrial corridors such as north Chennai, emissions from refining, thermal power "
        "generation, port freight, and urban traffic interact with sea-breeze circulations and seasonal monsoons to shape "
        "daily ambient pollutant concentrations."
    )
    add_body(
        doc,
        "Short- to medium-term forecasting of the composite Air Quality Index (AQI) provides environmental engineering teams "
        "with advance visibility into deteriorating dispersion conditions. Rather than reacting only after regulatory limits "
        "are exceeded, accurate 1-day and 7-day ambient forecasts support proactive operational planning, ambient risk "
        "communication, and schedule coordination around meteorological stagnation episodes."
    )

    doc.add_heading("1.2 Problem Statement and Objectives", level=2)
    add_body(doc, "This engineering design project addresses the following problem specification:")
    add_quote(
        doc,
        "\"Indian Oil Corporation seeks to forecast the Air Quality Index (AQI) around its refinery zones. Using historical "
        "pollutant levels, an ARIMA and LSTM-based forecasting model is needed to predict AQI values and support proactive "
        "emission control.\""
    )
    add_body(doc, "To fulfill this specification rigorously, five engineering objectives were established:")
    add_bullet(doc, "Objective 1 (Data Integrity & Site Selection): ",
               "Audit multi-station and city-level CPCB monitoring archives around the Chennai Petroleum Corporation Limited (CPCL) "
               "Manali refinery zone and construct a continuous, leakage-free daily modeling series.")
    add_bullet(doc, "Objective 2 (Domain-Faithful Preprocessing): ",
               "Preserve authentic high-pollution physical events (such as festival firecrackers and winter inversions) while isolating "
               "verified telemetry sensor malfunctions, applying gap-adaptive imputation and train-only scaling.")
    add_bullet(doc, "Objective 3 (Classical & Deep Learning Modeling): ",
               "Develop and tune both classical state-space models (ARIMA, ARIMA with annual Fourier harmonics, and exogenous ARIMAX) "
               "and recurrent neural networks (Stacked LSTM, GRU, BiLSTM, CNN-LSTM, direct 7-day LSTM, and an ARIMAX+LSTM hybrid).")
    add_bullet(doc, "Objective 4 (Rigorous Benchmarking & Uncertainty Quantification): ",
               "Compare all models across identical chronological partitions using regression accuracy, CPCB category classification, "
               "Harvey–Leybourne–Newbold corrected Diebold–Mariano tests, regulatory alert metrics, and calibrated prediction intervals.")
    add_bullet(doc, "Objective 5 (Structural-Break Stress Testing): ",
               "Evaluate model robustness without retraining during the 2020 COVID-19 national lockdown structural break (1 March to 30 June 2020).")

    doc.add_heading("1.3 Scope of the Report", level=2)
    add_body(
        doc,
        "This report documents the complete seven-phase engineering pipeline. Section 2 presents conceptual background on CPCB AQI, "
        "time-series modeling, and statistical evaluation. Section 3 details dataset provenance and the formal selection of Chennai "
        "city-level daily data. Section 4 outlines the end-to-end architecture and chronological leakage controls. Sections 5 and 6 "
        "report preprocessing and exploratory diagnostics. Sections 7 and 8 detail classical ARIMA/ARIMAX and deep learning model "
        "development. Section 9 synthesizes unified benchmark results, statistical tests, multi-step errors, intervals, and the "
        "lockdown stress test. Sections 10 through 12 examine modeling considerations, limitations, future scope, and final conclusions."
    )

    # =========================================================================
    # SECTION 2: BACKGROUND
    # =========================================================================
    doc.add_heading("2. Background", level=1)
    doc.add_heading("2.1 CPCB National Air Quality Index and Categories", level=2)
    add_body(
        doc,
        "The Central Pollution Control Board (CPCB) introduced India's National Air Quality Index in 2014 to translate complex "
        "multivariate pollutant concentrations into a single public-health communication scale ranging from 0 to 500. Under CPCB "
        "methodology, sub-indices are computed for up to eight pollutants (PM10, PM2.5, NO2, SO2, CO, O3, NH3, and Pb) using "
        "piecewise-linear breakpoint functions based on 24-hour averages (8-hour averages for CO and O3). The daily composite AQI "
        "is assigned as the maximum sub-index across monitored species, provided at least three pollutants including PM2.5 or PM10 "
        "are valid. The index maps onto six qualitative health categories: Good (0–50), Satisfactory (51–100), Moderate (101–200), "
        "Poor (201–300), Very Poor (301–400), and Severe (401–500)."
    )

    doc.add_heading("2.2 Classical Time Series: ARIMA, ARIMAX, and Fourier Seasonality", level=2)
    add_body(
        doc,
        "Autoregressive Integrated Moving Average (ARIMA(p, d, q)) models represent a univariate series after d orders of non-seasonal "
        "differencing as a linear combination of p lagged observations and q lagged forecast innovations. When external covariates such "
        "as lagged fine particulate matter PM2.5(t−1) are introduced into the observation equation, the formulation extends to ARIMAX. "
        "For daily environmental series exhibiting annual seasonality (period m = 365.25 days), classical seasonal ARIMA (SARIMA) becomes "
        "computationally intractable because seasonal differencing creates a 365-dimensional state-space transition matrix. An efficient "
        "alternative represents annual periodicity through K pairs of deterministic Fourier sine and cosine harmonics entered as exogenous "
        "regressors, capturing smooth seasonal cycles with only 2K parameters."
    )

    doc.add_heading("2.3 Recurrent Neural Networks: LSTM, GRU, and CNN-LSTM", level=2)
    add_body(
        doc,
        "Long Short-Term Memory (LSTM) networks address vanishing gradients in sequence modeling by regulating information flow across "
        "time steps via forget, input, and output gates paired with an additive cell state. Gated Recurrent Units (GRU) simplify this "
        "architecture by merging cell and hidden states through reset and update gates, reducing parameter counts on smaller datasets. "
        "A hybrid CNN-LSTM prepends a 1-D temporal convolutional layer before recurrent layers to extract local multi-pollutant motifs "
        "(such as multi-day stagnation ramps) prior to sequence aggregation. Training deep networks with Huber loss stabilizes gradients "
        "when occasional extreme pollution spikes introduce heavy right-tail residuals."
    )

    doc.add_heading("2.4 Forecasting Evaluation, Statistical Testing, and Conformal Intervals", level=2)
    add_body(
        doc,
        "Point forecast accuracy is evaluated using scale-dependent Root Mean Squared Error (RMSE, penalizing large spike errors) and "
        "Mean Absolute Error (MAE), relative Mean Absolute Percentage Error (MAPE), and Coefficient of Determination (R²). Category "
        "classification uses overall CPCB accuracy, Macro-F1 (accounting for rare-class imbalance), and Cohen's Quadratic Weighted Kappa "
        "(QWK), which penalizes multi-category misclassifications quadratically. Predictive accuracy differences between two competing "
        "models are assessed formally via the Diebold–Mariano (DM) test with the Harvey–Leybourne–Newbold (HLN) small-sample correction, "
        "supplemented by Bonferroni adjustment when conducting multiple pairwise tests. Finally, prediction uncertainty is quantified via "
        "Monte Carlo (MC) dropout for epistemic parameter spread and split-conformal prediction, which calibrates distribution-free residual "
        "quantiles on held-out validation errors."
    )

    # =========================================================================
    # SECTION 3: DATASET
    # =========================================================================
    doc.add_heading("3. Dataset", level=1)
    doc.add_heading("3.1 Data Source and Granularity Selection", level=2)
    add_body(
        doc,
        "The empirical dataset originates from the public Kaggle archive 'Air Quality Data in India (2015–2020)' compiled by Vopani "
        "from official CPCB Continuous Ambient Air Quality Monitoring Stations (CAAQMS). Although the archive provides both hourly and "
        "daily files (city_hour.csv, station_hour.csv, city_day.csv, station_day.csv, and stations.csv), this project strictly utilizes "
        "daily aggregated records. Daily granularity aligns directly with 24-hour CPCB regulatory reporting intervals, suppresses "
        "sub-daily rush-hour boundary-layer volatility, stabilizes linear state-space estimation, and yields tractable sequence lengths "
        "for deep neural training."
    )

    doc.add_heading("3.2 Location Rationale and Formal Series Selection Rule", level=2)
    add_body(
        doc,
        "Chennai was selected as the primary geographic target because Indian Oil Corporation's subsidiary, Chennai Petroleum Corporation "
        "Limited (CPCL), operates its major 10.5 MMTPA Manali Refinery complex in north Chennai. Within Chennai, CPCB metadata lists four "
        "monitoring stations: TN001 (Alandur Bus Depot), TN002 (Manali Village), TN003 (Manali CPCB), and TN004 (Velachery Residential Area). "
        "Two stations (TN002 and TN003) sit inside the Manali industrial corridor."
    )
    add_body(
        doc,
        "To prevent subjective selection, an explicit quantitative data-quality rule was enforced prior to modeling over the pre-lockdown "
        "audit window (2015-01-01 to 2020-02-29, 1,886 calendar days): a Manali station would be selected if and only if it contained "
        "(1) at least 3 years of historical depth (>= 1,095 days), (2) <= 20% missing AQI, and (3) no continuous missing gap exceeding 30 days. "
        "Otherwise, the Chennai city-level daily series in city_day.csv would be used. Table 3.1 summarizes the audit outcome."
    )

    add_styled_table(
        doc, "3.1",
        "Pre-COVID Candidate Series Comparison (2015-01-01 to 2020-02-29)",
        ["Candidate Series", "Span (Days)", "Valid Days", "Missing AQI (%)", "Max Gap (Days)", "Gaps >30d", "Usable Pollutants"],
        [
            ["Chennai City-Level (Selected)", "1,886", "1,761", "6.63%", "82", "1", "10"],
            ["Manali CPCB (Station TN003)", "1,886", "1,296", "31.28%", "411", "2", "7"],
            ["Manali Village (Station TN002)", "256", "218", "14.84%", "26", "0", "11"],
            ["Alandur Bus Depot (Station TN001)", "1,886", "1,541", "18.29%", "149", "2", "9"],
        ],
        col_widths=[2.1, 0.75, 0.75, 0.9, 0.8, 0.65, 0.85],
        highlight_rows={0}
    )

    add_body(
        doc,
        "Station TN002 contained only 256 days (beginning June 2019), failing the 3-year historical depth criterion. Station TN003 "
        "suffered 31.28% missingness and two extended outages (411 days in 2016–2017 and 136 days in 2015), failing both the missingness "
        "and gap-length rules. Consequently, the Chennai city-level series (city_day.csv) was formally selected. Furthermore, calendar "
        "year 2015 was excluded entirely because it contained an 82-day monitoring outage (September–November 2015). Starting the study on "
        "2016-01-01 through 2020-06-30 yields 1,643 continuous calendar days with near-zero post-2015 outages."
    )

    doc.add_heading("3.3 Pollutant Screening and Feature Sets", level=2)
    add_body(
        doc,
        "Twelve candidate pollutants in the pre-lockdown Chennai series were screened against a 30% missingness ceiling. PM10 (87.86% missing) "
        "and Xylene (100.00% missing) were excluded. Among the ten remaining species, NH3 exhibited 24.50% missingness (including a 411-day "
        "early gap) and near-zero Pearson correlation with AQI (r = 0.0075), leaving nine consistently recorded features (ALL9: PM2.5, NO, "
        "NO2, NOx, CO, SO2, O3, Benzene, Toluene). Because PM2.5 (6.20% missing, r = 0.6741 in full-span screening), CO (1.33% missing, r = 0.3515), "
        "O3 (2.60% missing, r = 0.2059), NO2 (1.91% missing), and SO2 (1.91% missing) directly represent primary combustion and refinery-relevant "
        "species, these five formed the primary CORE feature set for multivariate deep learning."
    )

    # =========================================================================
    # SECTION 4: METHODOLOGY
    # =========================================================================
    doc.add_heading("4. Methodology", level=1)
    doc.add_heading("4.1 Overall System Architecture", level=2)
    add_body(
        doc,
        "The forecasting framework follows a strict chronological pipeline progressing from data ingestion and auditing through domain-aware "
        "preprocessing, exploratory diagnostics, parallel classical and deep learning modeling tracks, unified multi-metric benchmarking, "
        "and structural-break stress testing. Figure 4.1 illustrates the complete end-to-end workflow."
    )
    add_figure(
        doc, "outputs/figures/flow_diagram.png", "4.1",
        "End-to-end AQI forecasting pipeline illustrating sequential preprocessing, exploratory diagnostics, parallel ARIMA/LSTM tracks, and evaluation.",
        width_in=6.2
    )

    doc.add_heading("4.2 Chronological Partitioning and Stress-Test Design", level=2)
    add_body(
        doc,
        "To eliminate temporal data leakage and preserve real-world forecasting deployment conditions, random cross-validation shuffling was "
        "strictly prohibited. The 1,643-day series was partitioned into four non-overlapping chronological blocks summarized in Table 4.1. "
        "The primary modeling series terminates on 2020-02-29, isolating the nationwide COVID-19 lockdown period (2020-03-01 to 2020-06-30) "
        "as an untouched out-of-distribution stress test."
    )

    add_styled_table(
        doc, "4.1",
        "Authoritative Dataset Partitions and Summary Statistics (FINAL_NUMBERS.md)",
        ["Split Name", "Start Date", "End Date", "Days (N)", "Mean AQI", "Std AQI", "Primary Function"],
        [
            ["Training", "2016-01-01", "2018-12-31", "1,096", "112.86", "46.91", "Model fitting, parameter & scaler calibration"],
            ["Validation", "2019-01-01", "2019-08-31", "243", "103.42", "36.72", "HPO, early stopping, conformal calibration"],
            ["Test", "2019-09-01", "2020-02-29", "182", "95.73", "45.03", "Held-out benchmarking & DM statistical tests"],
            ["Stress (Lockdown)", "2020-03-01", "2020-06-30", "122", "78.81", "20.38", "Out-of-distribution structural break testing"],
            ["Full Series", "2016-01-01", "2020-06-30", "1,643", "107.04", "44.92", "Complete continuous audited series"],
        ],
        col_widths=[1.25, 0.9, 0.9, 0.65, 0.7, 0.65, 1.85],
        highlight_rows={2, 3}
    )

    doc.add_heading("4.3 Strict Leakage Controls", level=2)
    add_body(doc, "Four strict temporal isolation safeguards governed every modeling phase:")
    add_bullet(doc, "Train-Only Parameter Fitting: ",
               "Seasonal imputation climatologies, 99.5th-percentile upper winsorization thresholds, and MinMaxScaler(0, 1) bounds were fitted "
               "exclusively on the 2016–2018 Training partition and applied downstream without refitting on Validation, Test, or Stress data.")
    add_bullet(doc, "Target-Timestamp Window Partitioning: ",
               "In sequence window generation (src/sequences.py), sliding input windows [t−L+1, ..., t] were partitioned strictly by their target "
               "timestamp (t+1 or t+1..t+7), allowing lookback context across split boundaries while preventing any future information leak.")
    add_bullet(doc, "Validation-Only Selection: ",
               "Grid searches, architecture tournaments, Optuna trials, conformal calibration quantiles, and Direct 7-Day LSTM training epochs "
               "were selected strictly on Validation errors before ever evaluating the Test split.")
    add_bullet(doc, "Identical Multi-Step Origins: ",
               "All multi-horizon models (Direct 7-Day LSTM, Recursive ARIMA-Fourier, Persistence) were evaluated on the identical set of 175 "
               "Test origins (2019-09-01 to 2020-02-22) where all seven forward horizons lie inside the Test partition.")

    # =========================================================================
    # SECTION 5: DATA PREPROCESSING
    # =========================================================================
    doc.add_heading("5. Data Preprocessing", level=1)
    doc.add_heading("5.1 Ordered Preprocessing Pipeline", level=2)
    add_body(
        doc,
        "Data preparation was implemented reproducibly in notebooks/03_preprocessing.ipynb across eleven deterministic steps: "
        "(1) loading city_day.csv, filtering Chennai, and clipping dates to 2016-01-01 through 2020-06-30 (1,643 days); "
        "(2) reindexing to a continuous daily DatetimeIndex; (3) physical bounds validation (confirming 0 negative values or CPCB >500 "
        "violations across all columns); (4) telemetry sensor anomaly removal; (5) split-wise missingness audit; (6) gap-adaptive imputation; "
        "(7) strict calendar-attributed spike inspection and upper-only winsorization; (8) cyclical calendar feature encoding; "
        "(9) chronological split partitioning; (10) train-isolated MinMaxScaler(0, 1) fitting and export; and (11) persistence of clean "
        "unscaled (for ARIMA) and scaled (for LSTM) CSV splits."
    )

    doc.add_heading("5.2 Sensor Anomaly Removal: Station TN004 Telemetry Malfunction", level=2)
    add_body(
        doc,
        "Exploratory cross-station verification uncovered an isolated telemetry anomaly in August–September 2016 where Station TN004 "
        "(Velachery) logged extreme CO spikes (up to 17–30 mg/m³ at station level, city-aggregated CO 2.08–11.03 mg/m³) driving reported "
        "city AQI to 215–449 while contemporaneous fine particulate matter PM2.5 remained in the Good range (12–23 ug/m³) and Station TN001 "
        "reported AQI near 64. Cross-checking station_day.csv verified that city AQI on 19–22 September 2016 (AQI 372, 437, 449, 434) and "
        "11 preceding days between 2016-08-15 and 2016-09-18 (AQI 215–266) was driven by the same TN004 CO sensor fault. Consequently, AQI "
        "and CO for all 15 confirmed anomaly days were set to NaN, flagged with anomaly_removed=1, and re-imputed. This reduced the maximum "
        "target AQI from the artificial 449.00 down to the genuine Diwali 2017 peak of 431.00."
    )

    doc.add_heading("5.3 Missingness Audit and Gap-Adaptive Imputation", level=2)
    add_body(
        doc,
        "After anomaly removal, missingness was confined strictly to the Training split (2016–2018), where AQI was missing on 4.65% of days "
        "(51 days across 14 gaps, max gap length 17 days), PM2.5 on 3.01% (33 days, max gap 17 days), CO on 3.38% (37 days, max gap 11 days), "
        "O3 on 3.47%, NO2 and SO2 on 3.01% each, Benzene on 4.56%, and Toluene on 4.38%. Validation, Test, and Stress splits all had 0.00% "
        "missingness across AQI and all CORE/ALL9 pollutants. Short gaps (<= 3 consecutive days) were filled by linear interpolation; medium "
        "gaps (4–30 days) were imputed using level-adjusted Training day-of-year (DOY) median climatologies. As shown in Figure 5.1, zero "
        "missing values remained post-imputation while preserving natural seasonal trajectories."
    )
    add_figure(
        doc, "outputs/figures/p3_imputation_before_after.png", "5.1",
        "Time series overlay comparing raw vs imputed values across short and medium monitoring gaps.",
        width_in=6.05
    )

    doc.add_heading("5.4 Authentic Spike Preservation and Upper-Only Winsorization", level=2)
    add_body(
        doc,
        "Target AQI spikes were explicitly NOT clipped or removed as outliers because extreme values represent genuine public-health hazards "
        "(including Diwali 2017 at AQI 431.0 and Bhogi 2017 at AQI 360.0). For pollutant inputs, upper-bound-only winsorization was applied "
        "using the Training 99.5th percentile threshold (Table 5.1). Lower clipping was intentionally omitted so clean monsoon scavenging "
        "readings and the 2020 lockdown drop remained unattenuated. Figure 5.2 illustrates the resulting upper-bound feature distributions, "
        "and Figure 5.3 displays the final cleaned AQI series across all four chronological partitions."
    )

    add_styled_table(
        doc, "5.1",
        "Upper-Bound-Only Winsorization Thresholds and Clipping Counts (Train 99.5th Percentile)",
        ["Pollutant", "Train 99.5th Percentile", "Total Clipped Days (of 1,643)", "Clipped Share (%)"],
        [
            ["PM2.5", "182.114 ug/m³", "7", "0.43%"],
            ["NO", "26.575 ug/m³", "13", "0.79%"],
            ["NO2", "40.212 ug/m³", "8", "0.49%"],
            ["NOx", "40.788 ppb", "18", "1.10%"],
            ["CO", "3.033 mg/m³", "6", "0.37%"],
            ["SO2", "28.769 ug/m³", "7", "0.43%"],
            ["O3", "72.189 ug/m³", "11", "0.67%"],
            ["Benzene", "12.315 ug/m³", "8", "0.49%"],
            ["Toluene", "18.258 ug/m³", "7", "0.43%"],
        ],
        col_widths=[1.5, 1.8, 1.8, 1.4]
    )

    add_figure(
        doc, "outputs/figures/p3_outlier_boxplots_clipping.png", "5.2",
        "Pollutant feature distributions showing upper-bound winsorization at the Training 99.5th percentile.",
        width_in=6.05
    )
    add_figure(
        doc, "outputs/figures/p3_final_cleaned_aqi_splits.png", "5.3",
        "Final cleaned daily Chennai AQI series (2016–2020) partitioned into Train, Validation, Test, and Stress windows.",
        width_in=6.05
    )

    # =========================================================================
    # SECTION 6: EXPLORATORY DATA ANALYSIS & VISUALIZATION
    # =========================================================================
    doc.add_heading("6. Exploratory Data Analysis & Visualization", level=1)
    doc.add_heading("6.1 Long-Term Trend, Seasonality, and Category Distribution", level=2)
    add_body(
        doc,
        "Exploratory analysis on the cleaned series reveals a multi-year downward drift in ambient AQI prior to the 2020 lockdown: "
        "Training (2016–2018) averaged 112.86 ± 46.91 (median 99.0, skewness 1.64), Validation (Jan–Aug 2019) averaged 103.42 ± 36.72, "
        "and Test (Sep 2019–Feb 2020) averaged 95.73 ± 45.03 (Table 4.1). Calendar year 2016 exhibited substantially higher mean (128.71) "
        "and volatility (std 45.16) than 2017–2019 (annual means 102.94–105.49), reflecting an ordinary least-squares trend slope of "
        "-7.616 AQI/year across annual means."
    )
    add_body(
        doc,
        "Seasonally, northeast winter (Nov–Jan) records the highest ambient levels (mean 126.67 ± 49.83) due to lower mixing heights and "
        "festival emissions, whereas the southwest monsoon (Jun–Aug) exhibits wet scavenging lows (96.30 ± 32.47). Conversely, day-of-week "
        "averages remain relatively flat (Mon 106.49 to Sat 116.13, Sun 112.77; cross-day std ~3.34 AQI), indicating the absence of a strong weekend effect. "
        "Regarding CPCB health bands, over 93% of days fall within Satisfactory (51–100) or Moderate (101–200), while Good (<=50), Very Poor "
        "(301–400), and Severe (>400) together account for under 1.5% of observations, creating strong class imbalance for multi-class evaluation."
    )

    doc.add_heading("6.2 Pollutant Correlations and STL Decomposition", level=2)
    add_body(
        doc,
        "Within the Training partition, PM2.5 exhibits the strongest contemporaneous correlation with AQI (Pearson r = 0.766, Spearman "
        "rho = 0.731; Figure 6.1) and the strongest next-day predictive cross-correlation (r = 0.729 against AQI(t+1)), followed by O3 "
        "(0.286) and CO (0.103). By contrast, NO2 (0.016) and SO2 (0.046) exhibit weak direct linear cross-correlation with next-day AQI. "
        "Lagged cross-correlations decay steadily beyond lags 3 to 7, directly motivating a compact lookback window for sequence models. "
        "Additive STL decomposition (Figure 6.2) quantifies annual seasonal strength at Fs = 0.5672 (m = 365), whereas weekly seasonal "
        "strength is Fs = 0.0079 (m = 7), justifying the omission of weekly seasonal differencing."
    )
    add_figure(
        doc, "outputs/figures/p4_pollutant_correlations.png", "6.1",
        "Spearman correlation matrix across AQI and criteria pollutant concentrations on the Training split.",
        width_in=6.0
    )
    add_figure(
        doc, "outputs/figures/p4_stl_decomposition.png", "6.2",
        "STL decomposition of Chennai daily AQI separating long-term trend, annual seasonal cycle, and residual spikes.",
        width_in=6.05
    )

    doc.add_heading("6.3 Stationarity Diagnostics and ACF / PACF Profile", level=2)
    add_body(
        doc,
        "Augmented Dickey–Fuller (ADF) and Kwiatkowski–Phillips–Schmidt–Shin (KPSS) tests were applied to Training AQI (Table 6.1). "
        "Raw AQI yields conflicting evidence: ADF rejects a unit root (stat = -5.6682, p = 9.05e-7), whereas KPSS rejects level "
        "stationarity (stat = 1.1417, p = 0.01) due to the multi-year trend shift. After first differencing (d=1), both tests agree on "
        "stationarity (ADF p = 1.99e-27; KPSS stat = 0.0856, p = 0.10)."
    )

    add_styled_table(
        doc, "6.1",
        "Stationarity Test Diagnostics on Training AQI (2016–2018, N=1,096)",
        ["Series", "ADF Statistic", "ADF p-Value", "KPSS Statistic", "KPSS p-Value", "Combined Verdict"],
        [
            ["Raw AQI (d = 0)", "-5.6682", "9.05e-07 (Reject H0)", "1.1417", "0.010 (Reject H0)", "Trend-nonstationary (conflicting)"],
            ["First-Differenced AQI (d = 1)", "-14.8172", "1.99e-27 (Reject H0)", "0.0856", "0.100 (Fail to reject)", "Strictly Stationary"],
        ],
        col_widths=[1.8, 0.9, 1.25, 0.9, 1.15, 1.4],
        highlight_rows={1}
    )

    add_body(
        doc,
        "The ACF and PACF profiles (Figure 6.3) show a sharp PACF cutoff after lag 1 (0.678 on raw AQI), suggesting low-order AR dynamics. "
        "Upon differencing (d=1), negative ACF spikes appear at lags 1 and 2 (-0.152, -0.170) alongside negative PACF across lags 1–9—an "
        "over-differencing signature—while daily lag-365 autocorrelation is r = 0.0816 (band ±0.0592), slightly above the 95% band; "
        "yearly seasonality is weak in the raw ACF but clear in STL and monthly means. "
        "Accordingly, both d=0 (stationary with deterministic Fourier harmonics) and d=1 were systematically tested during Phase 5 grid search."
    )
    add_figure(
        doc, "outputs/figures/p4_acf_pacf.png", "6.3",
        "ACF and PACF correlograms of raw (d=0) and first-differenced (d=1) Chennai daily AQI.",
        width_in=6.05
    )

    # =========================================================================
    # SECTION 7: ARIMA / ARIMAX MODELLING
    # =========================================================================
    doc.add_heading("7. ARIMA / ARIMAX Modelling", level=1)
    doc.add_heading("7.1 Baselines and Systematic Grid Search", level=2)
    add_body(
        doc,
        "Three non-parametric baselines anchored model evaluation: Naive Daily Persistence (AQI_hat(t+1) = AQI(t)), 7-Day Moving Average, "
        "and 365-Day Seasonal Naive. Across 80+ classical configurations in outputs/results/p5_grid_search.csv, three incremental families "
        "were tested using orders p, q in {0..3} and d in {0, 1}: M1 (Pure ARIMA), M2 (ARIMA + annual Fourier pairs K in {1, 2, 3}), and "
        "M3 (ARIMAX + Fourier + lagged exogenous PM2.5(t−1)). All models were ranked strictly by Validation RMSE (Table 7.1)."
    )

    add_styled_table(
        doc, "7.1",
        "Validation Comparison Across Differencing Regimes (d=0 vs d=1) and Model Families",
        ["Model Family & Specification", "Differencing", "Fourier K", "Exog PM2.5", "AIC", "Val RMSE", "Val MAE"],
        [
            ["M1 Best d=0: ARIMA(2,0,2)", "d = 0", "0", "No", "10,865.37", "30.441", "20.524"],
            ["M1 Best d=1: ARIMA(1,1,1)", "d = 1", "0", "No", "10,858.23", "30.381", "20.454"],
            ["M2 Best d=1: ARIMA(1,1,1)+F3", "d = 1", "3", "No", "10,853.91", "30.124", "19.966"],
            ["M3 Best d=0: ARIMAX(1,0,0)+F3+PM2.5", "d = 0", "3", "Yes", "10,601.14", "30.128", "18.164"],
            ["M3 Best d=1: ARIMAX(2,1,1)+F3+PM2.5", "d = 1", "3", "Yes", "10,533.66", "28.907", "17.333"],
        ],
        col_widths=[2.1, 0.75, 0.7, 0.75, 0.85, 0.8, 0.75],
        highlight_rows={4}
    )

    add_body(
        doc,
        "Critical AIC Caveat across Differencing Orders: Raw Akaike Information Criterion (AIC) values are mathematically non-comparable "
        "between d=0 and d=1 models because differencing alters both the effective sample size (N−d) and the probability Jacobian of the "
        "target variable. Therefore, model selection between d=0 and d=1 relied strictly on out-of-sample Validation RMSE and MAE rather than "
        "AIC. Comparing best candidates shows that d=1 outperforms d=0 (Val RMSE 28.907 vs 30.128, Val MAE 17.333 vs 18.164), consistent with "
        "first-differencing tracking multi-day local level drift more responsively than a global constant mean."
    )

    doc.add_heading("7.2 Exogenous Fourier Harmonics, PM2.5, and Training-Window Sensitivity", level=2)
    add_body(
        doc,
        "Incorporating K=3 annual Fourier harmonic pairs (6 sinusoidal terms at period 365.25 days) reduced validation error smoothly relative "
        "to Pure ARIMA, and adding 1-day-lagged PM2.5(t−1) yielded a further sharp improvement (Val RMSE 30.124 -> 28.907). To verify whether "
        "including 2016 (which exhibited higher mean AQI 128.71) helped or hurt 2019 calibration, a training-window sensitivity check compared "
        "the full 2016–2018 window (N=1,096 days) against a truncated 2017–2018 window (N=730 days). Retaining 2016 improved Validation RMSE "
        "from 29.920 to 28.907 (Val MAE 18.665 -> 17.333), consistent with needing three complete annual cycles to calibrate K=3 seasonal harmonics."
    )

    doc.add_heading("7.3 Residual Diagnostics, Walk-Forward Evaluation, and Multi-Step Results", level=2)
    add_body(
        doc,
        "Residual diagnostics on the fitted ARIMAX(2,1,1)+F3+PM2.5 model (Figure 7.1) yield a near-zero mean (-0.273, std 29.248) with "
        "positive skewness (1.384) and high kurtosis (10.868) caused by unmodeled episodic spikes. Ljung–Box independence tests do not reject "
        "white noise at lag 10 (Q = 10.61, p = 0.3889) or lag 20 (Q = 17.70, p = 0.6073). Evaluated via fixed-parameter walk-forward forecasting "
        "(Kalman filtering without refitting) over the 182-day Test set (Figure 7.2), ARIMAX achieved Test RMSE 25.97, MAE 18.45, R² 0.6657, "
        "and CPCB accuracy 76.92% (versus Pure ARIMA Test RMSE 30.58 and Persistence 32.10). For 7-day recursive forecasting without future "
        "measured PM2.5, ARIMA-Fourier was unrolled across the 175 Test origins, achieving RMSE 32.25 (h=1), 42.83 (h=2), 46.05 (h=3), "
        "47.83 (h=4), 49.33 (h=5), 50.06 (h=6), and 50.35 (h=7), outperforming Persistence (32.72 to 61.16) at every horizon."
    )
    add_figure(
        doc, "outputs/figures/p5_model_diagnostics.png", "7.1",
        "Four-panel residual diagnostic suite for ARIMAX(2,1,1)+Fourier(K=3)+PM2.5 confirming white-noise residuals with heavy positive tails.",
        width_in=6.15
    )
    add_figure(
        doc, "outputs/figures/p5_test_actual_vs_predicted.png", "7.2",
        "ARIMAX 1-day walk-forward forecast vs actual AQI on the Test partition with 95% parametric prediction intervals.",
        width_in=6.15
    )

    # =========================================================================
    # SECTION 8: LSTM / DEEP LEARNING MODELLING
    # =========================================================================
    doc.add_heading("8. LSTM / Deep Learning Modelling", level=1)
    doc.add_heading("8.1 Sequence Construction and Staged Exploration (Stages 1–3)", level=2)
    add_body(
        doc,
        "Deep learning experiments were executed on CPU across 5 deterministic random seeds (42, 123, 456, 789, 1011) using Train-fitted "
        "MinMaxScaler(0, 1) inputs (src/train_dl.py). Sliding windows of shape (L, D) were strictly partitioned by target date. A three-stage "
        "search avoided unguided over-parameterization:"
    )
    add_bullet(doc, "Stage 1 — Feature Set x Lookback Factorial Grid (80 runs): ",
               "Using a baseline 2-layer Stacked LSTM (64 units, dropout 0.2, Adam lr=1e-3, MSE), four feature sets (Univariate U, Parsimonious P, "
               "CORE C, ALL9 A) were crossed with four lookbacks L in {7, 14, 30, 60}. Compact L=7 with CORE5 (D=8 including AQI and sin/cos DOY) "
               "achieved the lowest 5-seed Validation RMSE of 27.375 +/- 0.513 (vs CORE5 L=14 at 27.906, Parsimonious L=7 at 28.177, ALL9 L=7 at 28.195, and Univariate L=7 "
               "at 29.765).")
    add_bullet(doc, "Stage 2 — Architecture Tournament (40 runs): ",
               "On the top two Stage 1 configurations (C, L=7 and C, L=14), four recurrent topologies were compared across 5 seeds (Figure 8.1). "
               "CNN-LSTM (Conv1D followed by LSTM layers) won with Validation RMSE 26.241 +/- 1.043 on CORE5 (L=7), outperforming Stacked GRU (27.148 at L=14, 27.207 at L=7), "
               "Stacked LSTM (27.375), and BiLSTM (27.427).")
    add_bullet(doc, "Stage 3 — Optuna Bayesian HPO (60 trials): ",
               "Tree-structured Parzen Estimator (TPE) optimization with Median Pruning searched layer units, dropout, recurrent dropout, learning "
               "rate, batch size, L2 regularization, and loss function (MSE vs Huber). Trial #10 achieved 2-seed Val RMSE 24.850 using Huber loss, "
               "batch size 16, lr = 0.00438, units (80, 48), dropout 0.15, and recurrent dropout 0.20 (learning rate accounted for 67.9% of HPO "
               "importance). Five-seed confirmation yielded Val RMSE 25.496 +/- 0.994 at a median of 18 epochs.")

    add_figure(
        doc, "outputs/figures/p6_stage2_architecture_comparison.png", "8.1",
        "Stage 2 deep learning architecture tournament comparing CNN-LSTM, GRU, BiLSTM, and Stacked LSTM across 5 seeds.",
        width_in=6.0
    )

    doc.add_heading("8.2 Final Retraining, MC Dropout, 7-Day Direct LSTM, and Hybrid Model", level=2)
    add_body(
        doc,
        "In Stage 4, the winning CNN-LSTM configuration was retrained on Train+Validation (1,339 days) across 5 seeds for 18 epochs, "
        "forming a 5-seed ensemble that achieved Test RMSE 27.63, MAE 21.21, R² 0.6214, and CPCB accuracy 68.68%. A standalone Univariate "
        "Stacked LSTM (L=7) benchmarked separately yielded Test RMSE 31.73 and MAE 22.87. Using 100 stochastic forward passes per seed (500 total), "
        "MC Dropout produced narrow 95% bands (mean width 31.69 AQI) that undercovered Test observations at 37.91%, reflecting epistemic "
        "parameter uncertainty only rather than full observation noise."
    )
    add_body(
        doc,
        "In Stage 5, a Direct Multi-Step CNN-LSTM with a Dense(7) output head was trained to emit h=1..7 simultaneously. Crucially, to prevent "
        "any test-set leakage, the retraining epoch count was determined strictly by evaluating Validation RMSE across all 7 horizons epoch-by-epoch "
        "across 5 seeds. Validation error reached its minimum at Epoch 2 (mean Val RMSE = 33.88) before rising rapidly due to the level shift between "
        "2016–2018 (mean 112.86) and 2019 (mean 103.42). Retrained on Train+Val for 2 epochs across 5 seeds (src/retrain_7day.py), the Direct 7-Day "
        "LSTM achieved RMSE 40.39 (h=1), 47.30 (h=2), 51.46 (h=3), 54.42 (h=4), 56.08 (h=5), 54.48 (h=6), and 53.04 (h=7). Finally, in Stage 6, a "
        "Residual Hybrid model (trained on L=7 lagged ARIMAX errors plus pollutants) improved standalone DL Test RMSE from 27.63 down to 26.81 "
        "(MAE 19.91, R² 0.6435)."
    )

    # =========================================================================
    # SECTION 9: RESULTS & COMPARISON
    # =========================================================================
    doc.add_heading("9. Results & Comparison", level=1)
    doc.add_heading("9.1 Master Test Benchmark and 2x2 Paradigm Grid", level=2)
    add_body(
        doc,
        "Table 9.1 consolidates 1-day-ahead performance across all eight evaluated models on the 182-day Test period (visualized in Figure 9.1). "
        "ARIMAX(2,1,1)+F3+PM2.5 achieved the strongest score on every metric except MAPE (where Pure ARIMA's slightly smaller low-end predictions "
        "gave 19.33% vs 20.25%), attaining Test RMSE 25.97, MAE 18.45, R² 0.6657, CPCB accuracy 76.92%, and QWK 0.6471."
    )

    add_styled_table(
        doc, "9.1",
        "Master 1-Day-Ahead Benchmark Results on Test Split (2019-09-01 to 2020-02-29, N=182 Days)",
        ["Model Family", "Specific Model", "RMSE", "MAE", "MAPE (%)", "R²", "CPCB Acc", "Macro F1", "QWK"],
        [
            ["Classical Champion", "ARIMAX(2,1,1)+F3+PM2.5", "25.97", "18.45", "20.25", "0.6657", "76.92%", "0.3664", "0.6471"],
            ["Ensemble Hybrid", "Hybrid ARIMAX + Residual LSTM", "26.81", "19.91", "22.58", "0.6435", "74.73%", "0.3644", "0.6296"],
            ["DL Champion", "CNN-LSTM (L=7, CORE)", "27.63", "21.21", "23.92", "0.6214", "68.68%", "0.3167", "0.5384"],
            ["Classical", "Pure ARIMA(2,1,1)", "30.58", "19.23", "19.33", "0.5362", "74.18%", "0.3473", "0.5587"],
            ["Deep Learning", "Univariate Stacked LSTM (L=7)", "31.73", "22.87", "25.43", "0.5008", "70.88%", "0.3205", "0.5226"],
            ["Baseline", "Naive Daily Persistence (182d)", "32.10", "19.56", "19.40", "0.4889", "73.63%", "0.3396", "0.5951"],
            ["Baseline", "7-Day Moving Average", "42.44", "26.49", "27.19", "0.1067", "64.84%", "0.2465", "0.3405"],
            ["Baseline", "Seasonal Naive (365-Day Lag)", "71.61", "53.86", "61.80", "-1.5426", "36.81%", "0.1179", "-0.0554"],
        ],
        col_widths=[1.1, 1.75, 0.55, 0.55, 0.6, 0.6, 0.65, 0.6, 0.55],
        highlight_rows={0}
    )

    add_figure(
        doc, "outputs/figures/p7_master_benchmark_bar.png", "9.1",
        "Master Test period benchmark comparing RMSE and MAE across classical, deep learning, hybrid, and baseline models.",
        width_in=6.15
    )

    add_body(
        doc,
        "Disentangling algorithmic paradigm from information scope in a balanced 2x2 factorial grid (Table 9.2, Figure 9.2) reveals consistent "
        "benefits from exogenous pollutant data: adding PM2.5 to classical ARIMA reduces Test RMSE by 15.1% (30.58 -> 25.97), while adding CORE "
        "pollutants to LSTM reduces Test RMSE by 12.9% (31.73 -> 27.63). Across both univariate and multivariate scopes, classical state-space "
        "models achieve lower Test RMSE than their deep learning counterparts (-3.6% univariate; -6.0% multivariate)."
    )

    add_styled_table(
        doc, "9.2",
        "2x2 Factorial Grid: Algorithmic Paradigm vs Information Scope (Test RMSE)",
        ["Algorithmic Paradigm", "Univariate (AQI Only)", "Multivariate (AQI + Pollutants)", "Multivariate Gain"],
        [
            ["Classical (ARIMA Family)", "30.58 (Pure ARIMA)", "25.97 (ARIMAX+F3+PM2.5)", "-4.61 (-15.1%)"],
            ["Deep Learning (LSTM Family)", "31.73 (Univariate LSTM)", "27.63 (CNN-LSTM, CORE)", "-4.10 (-12.9%)"],
            ["Classical Advantage", "-1.15 (-3.6%)", "-1.66 (-6.0%)", "—"],
        ],
        col_widths=[1.8, 1.8, 2.0, 1.3],
        highlight_rows={0}
    )
    add_figure(
        doc, "outputs/figures/p7_2x2_univariate_multivariate.png", "9.2",
        "2x2 factorial comparison showing Test RMSE improvements from exogenous pollutant inputs across both paradigms.",
        width_in=5.8
    )

    doc.add_heading("9.2 Diebold–Mariano Statistical Hypothesis Tests", level=2)
    add_body(
        doc,
        "Pairwise Diebold–Mariano tests with HLN small-sample correction under squared-error loss (h=1, T=182) are summarized in Table 9.3 and "
        "Figure 9.3. At an unadjusted alpha = 0.05 significance level, ARIMAX significantly outperforms Persistence (p = 0.0132), Pure ARIMA "
        "(p = 0.0297), CNN-LSTM (p = 0.0127), and Hybrid (p = 0.0001), whereas CNN-LSTM vs Persistence does not reach significance (p = 0.0758). "
        "However, when applying a conservative Bonferroni correction across the family of 10 pairwise tests (alpha_adj = 0.05 / 10 = 0.005), "
        "only the ARIMAX vs Hybrid comparison remains statistically significant (p = 0.0001)."
    )

    add_styled_table(
        doc, "9.3",
        "HLN-Corrected Diebold–Mariano Pairwise Hypothesis Tests (Test Period, T=182)",
        ["Comparison (M1 vs M2)", "DM Stat", "p-Value", "Sig. at Unadjusted alpha=0.05?", "Sig. at Bonferroni alpha=0.005?"],
        [
            ["ARIMAX vs Persistence", "-2.504", "0.0132", "Yes", "No"],
            ["ARIMAX vs Pure ARIMA", "-2.192", "0.0297", "Yes", "No"],
            ["ARIMAX vs CNN-LSTM", "-2.519", "0.0127", "Yes", "No"],
            ["ARIMAX vs Hybrid", "-3.935", "0.0001", "Yes", "Yes (p < 0.005)"],
            ["CNN-LSTM vs Persistence", "-1.786", "0.0758", "No", "No"],
            ["Pure ARIMA vs Persistence", "-0.861", "0.3903", "No", "No"],
            ["Hybrid vs Persistence", "-2.126", "0.0349", "Yes", "No"],
            ["Hybrid vs CNN-LSTM", "-1.284", "0.2010", "No", "No"],
            ["Hybrid vs Pure ARIMA", "-1.751", "0.0818", "No", "No"],
            ["CNN-LSTM vs Pure ARIMA", "-1.488", "0.1385", "No", "No"],
        ],
        col_widths=[1.9, 0.8, 0.8, 1.65, 1.75],
        highlight_rows={3}
    )
    add_figure(
        doc, "outputs/figures/p7_diebold_mariano_heatmap.png", "9.3",
        "Pairwise Diebold–Mariano HLN-corrected two-sided p-value heatmap across benchmark models.",
        width_in=5.8
    )

    doc.add_heading("9.3 Operational Alerts and Category Classification", level=2)
    add_body(
        doc,
        "Binary alert metrics were evaluated at CPCB action thresholds AQI > 100 (Moderate+, 52 actual Test days) and AQI > 200 (Poor+, 10 actual "
        "Test days), as displayed in Table 9.4 and Figure 9.4, alongside category confusion matrices in Figure 9.5. At AQI > 100, Hybrid achieved "
        "86.54% recall and 0.7031 F1, while ARIMAX achieved the highest precision (64.52%) with 76.92% recall and 0.7018 F1, clearly surpassing "
        "Persistence (F1 0.6476). At AQI > 200, ARIMAX and Hybrid tied at 85.71% precision, 60.00% recall, and 0.7059 F1, outperforming CNN-LSTM "
        "(recall 40.00%, F1 0.5333), though the 10-day sample (5.5% frequency) limits rare-event statistical power."
    )

    add_styled_table(
        doc, "9.4",
        "Operational Alert Evaluation on Test Split (N=182 Days)",
        ["Threshold & Metric", "ARIMAX", "Hybrid", "CNN-LSTM", "Pure ARIMA", "Persistence"],
        [
            ["AQI > 100 Precision (52 actual days)", "64.52%", "59.21%", "53.01%", "61.11%", "64.15%"],
            ["AQI > 100 Recall", "76.92%", "86.54%", "84.62%", "63.46%", "65.38%"],
            ["AQI > 100 F1 Score", "0.7018", "0.7031", "0.6519", "0.6226", "0.6476"],
            ["AQI > 200 Precision (10 actual days)", "85.71%", "85.71%", "80.00%", "100.00%", "70.00%"],
            ["AQI > 200 Recall", "60.00%", "60.00%", "40.00%", "50.00%", "70.00%"],
            ["AQI > 200 F1 Score", "0.7059", "0.7059", "0.5333", "0.6667", "0.7000"],
        ],
        col_widths=[2.4, 0.9, 0.9, 0.9, 0.9, 0.9],
        highlight_rows={2, 5}
    )
    add_figure(
        doc, "outputs/figures/p7_alert_precision_recall.png", "9.4",
        "Operational alert Precision, Recall, and F1 scores at AQI > 100 and AQI > 200 thresholds.",
        width_in=6.1
    )
    add_figure(
        doc, "outputs/figures/p7_side_by_side_confusion_matrices.png", "9.5",
        "CPCB qualitative health category confusion matrices for ARIMAX, CNN-LSTM, and Hybrid on the Test split.",
        width_in=6.2
    )

    doc.add_heading("9.4 Stratified Spike Error Analysis and Worst-Day Calendar Audit", level=2)
    add_body(
        doc,
        "Decomposing Test errors between normal days (bottom 90%, actual AQI < 134.9, N=163) and spike days (top 10%, AQI >= 134.9, N=19) "
        "highlights systematic negative bias on episodic peaks (Figure 9.6). While ARIMAX achieved 18.78 RMSE (+9.56 bias) on normal days, "
        "all models degraded sharply on top-10% spikes (mean negative bias ranging from -27.6 to -55.5 AQI). Table 9.5 audits the top 5 "
        "worst-forecast days: exactly 2 of the 5 worst days correspond to calendar-verified festival events (Diwali on 2019-10-27 and Bhogi on "
        "2020-01-14), while the remaining 3 days carry 'No matched event'. Figure 9.7 illustrates the Diwali 2019 under-prediction across all models."
    )

    add_styled_table(
        doc, "9.5",
        "Top 5 Worst-Forecast Test Days Audit (FINAL_NUMBERS.md)",
        ["Date", "Actual AQI", "ARIMAX Error", "CNN-LSTM Error", "Calendar Match", "Context"],
        [
            ["2019-10-27", "306.0", "-140.81", "-144.74", "Diwali 2019", "Verified festival pyrotechnics spike"],
            ["2020-01-14", "230.0", "-92.22", "-85.63", "Bhogi 2020", "Verified harvest bonfire smoke event"],
            ["2019-11-04", "222.0", "-90.43", "-83.72", "No matched event", "Sudden post-monsoon jump from 91 to 222 AQI"],
            ["2019-11-12", "77.0", "+72.90", "+66.91", "No matched event", "Over-prediction after rapid ambient drop"],
            ["2019-11-07", "273.0", "-67.96", "-75.09", "No matched event", "Unmodeled severe pollution buildup"],
        ],
        col_widths=[0.95, 0.8, 0.95, 1.05, 1.25, 1.9],
        highlight_rows={0, 1}
    )
    add_figure(
        doc, "outputs/figures/p7_spike_vs_normal_error.png", "9.6",
        "RMSE and MAE degradation between normal Test days (bottom 90%) and spike days (top 10%).",
        width_in=6.0
    )
    add_figure(
        doc, "outputs/figures/p7_test_overlay_zoomed.png", "9.7",
        "Test period forecast trajectory overlay comparing ARIMAX, CNN-LSTM, and Hybrid with zoomed Diwali 2019 inset.",
        width_in=6.15
    )

    doc.add_heading("9.5 Multi-Horizon Forecasting (h = 1 to 7 Days)", level=2)
    add_body(
        doc,
        "Evaluated across the 175 aligned Test origins (Table 9.6, Figure 9.8), the validation-selected Epoch-2 Direct Multi-Step LSTM beats "
        "Persistence only at horizons 5, 6, and 7 (56.08 vs 58.42; 54.48 vs 61.38; 53.04 vs 61.16), whereas both Persistence and recursive "
        "ARIMA-Fourier outperform it at h=1..4. Recursive ARIMA-Fourier achieves the lowest RMSE across every horizon (32.25 to 50.35), "
        "consistent with the hypothesis that deterministic annual Fourier harmonics constrain multi-day trajectory dispersion."
    )

    add_styled_table(
        doc, "9.6",
        "Multi-Horizon Forecast Benchmark Across 175 Identical Test Origins (h = 1 to 7 Days)",
        ["Horizon", "Persist RMSE", "ARIMA-F3 RMSE", "LSTM-7d RMSE", "Persist MAE", "ARIMA-F3 MAE", "LSTM-7d MAE", "LSTM Beats Persist?"],
        [
            ["h = 1", "32.72", "32.25", "40.39", "20.15", "22.37", "31.96", "No"],
            ["h = 2", "47.14", "42.83", "47.30", "28.99", "31.64", "37.45", "No"],
            ["h = 3", "51.01", "46.05", "51.46", "32.01", "35.01", "41.85", "No"],
            ["h = 4", "53.46", "47.83", "54.42", "33.79", "36.72", "44.91", "No"],
            ["h = 5", "58.42", "49.33", "56.08", "36.80", "38.07", "46.92", "Yes (56.08 < 58.42)"],
            ["h = 6", "61.38", "50.06", "54.48", "38.05", "38.87", "44.46", "Yes (54.48 < 61.38)"],
            ["h = 7", "61.16", "50.35", "53.04", "37.53", "39.34", "42.76", "Yes (53.04 < 61.16)"],
        ],
        col_widths=[0.65, 0.85, 0.95, 0.9, 0.8, 0.9, 0.85, 1.0],
        highlight_rows={4, 5, 6}
    )
    add_figure(
        doc, "outputs/figures/p7_rmse_vs_horizon_final.png", "9.8",
        "Multi-horizon RMSE growth from h=1 to h=7 days on the 175 aligned Test origins.",
        width_in=6.1
    )

    doc.add_heading("9.6 Uncertainty Quantification: Prediction Intervals", level=2)
    add_body(
        doc,
        "Split-conformal calibration using finite-sample Validation residual quantiles (q = 63.87 for ARIMAX, q = 57.71 for LSTM) achieved "
        "96.70% and 96.15% empirical coverage on Test (Table 9.7, Figure 9.9), and 98.36% on Stress, whereas MC Dropout covered only 37.91% on "
        "Test. Note that split-conformal theoretical marginal guarantees assume data exchangeability, which time series with level drift do not "
        "strictly satisfy; empirical coverage here therefore serves as an observed calibration check rather than a formal guarantee."
    )

    add_styled_table(
        doc, "9.7",
        "95% Prediction Interval Calibration Benchmark (Test N=182, Stress N=122)",
        ["Method", "Interval Type", "Val Quantile q", "Mean Width (AQI)", "Test Coverage", "Stress Coverage"],
        [
            ["ARIMAX Split-Conformal", "Distribution-free", "63.87", "127.74", "96.70%", "98.36%"],
            ["LSTM Split-Conformal", "Distribution-free", "57.71", "115.43", "96.15%", "98.36%"],
            ["ARIMAX Parametric", "Gaussian (+/-1.96 sigma)", "N/A", "114.39", "96.15%", "97.54%"],
            ["LSTM MC-Dropout", "Epistemic parameter", "N/A", "31.69", "37.91%", "N/A"],
        ],
        col_widths=[1.6, 1.4, 0.9, 1.05, 0.95, 1.0],
        highlight_rows={0, 1}
    )
    add_figure(
        doc, "outputs/figures/p7_interval_comparison.png", "9.9",
        "Comparison of Split-Conformal, Parametric Gaussian, and MC-Dropout prediction intervals over a 30-day Test window.",
        width_in=6.1
    )

    doc.add_heading("9.7 COVID-19 Lockdown Stress Test (2020-03-01 to 2020-06-30)", level=2)
    add_body(
        doc,
        "During the 122-day COVID-19 national lockdown structural break (mean AQI 78.81 vs pre-lockdown full-series mean 109.30 and Test mean "
        "95.73), Pure ARIMA achieved the lowest stress RMSE (13.51), followed by persistence (14.45); models using exogenous or seasonal terms "
        "(ARIMAX 17.74, multivariate CNN-LSTM 20.28) performed worst, with the LSTMs over-predicting (mean bias +7.84 / +10.55) and ARIMAX "
        "under-predicting (−8.32). Table 9.8 and Figure 9.10 summarize Validation vs Stress performance alongside mean forecast bias."
    )

    add_styled_table(
        doc, "9.8",
        "Validation (N=243) and COVID Lockdown Stress Window (N=122) Metrics with Stress Mean Bias",
        ["Model Name", "Val RMSE", "Val MAE", "Stress RMSE", "Stress MAE", "Stress R²", "Stress Acc", "Stress Bias"],
        [
            ["Pure ARIMA(2,1,1)", "30.49", "20.45", "13.51", "9.06", "0.5567", "86.89%", "-1.15"],
            ["Naive Persistence", "34.27", "20.85", "14.45", "9.41", "0.4928", "87.70%", "-0.31"],
            ["7-Day MA", "34.50", "24.27", "14.65", "10.22", "0.4793", "86.89%", "-0.82"],
            ["Hybrid ARIMAX+LSTM", "—", "—", "15.62", "9.46", "0.4080", "86.89%", "-3.19"],
            ["Univariate LSTM", "29.61", "20.03", "15.98", "12.77", "0.3801", "86.07%", "+7.84"],
            ["ARIMAX(2,1,1)+F3+PM2.5", "28.91", "17.33", "17.74", "12.37", "0.2361", "80.33%", "-8.32"],
            ["CNN-LSTM (Multivariate)", "25.63", "17.03", "20.28", "15.40", "0.0019", "81.97%", "+10.55"],
            ["Seasonal Naive (365d)", "46.33", "34.76", "37.93", "28.82", "-2.4913", "62.30%", "+22.70"],
        ],
        col_widths=[1.6, 0.65, 0.65, 0.75, 0.75, 0.7, 0.75, 0.75],
        highlight_rows={0, 1}
    )
    add_figure(
        doc, "outputs/figures/p7_stress_overlay.png", "9.10",
        "COVID lockdown stress window trajectories comparing Pure ARIMA, ARIMAX, CNN-LSTM, Hybrid, and Persistence against actual AQI.",
        width_in=6.15
    )

    # =========================================================================
    # SECTION 10: DISCUSSION
    # =========================================================================
    doc.add_heading("10. Discussion", level=1)
    doc.add_heading("10.1 Why ARIMAX Outperformed CNN-LSTM on This Dataset", level=2)
    add_body(
        doc,
        "A central empirical outcome of this study is that classical ARIMAX(2,1,1)+F3+PM2.5 outperformed the tuned CNN-LSTM on the held-out "
        "Test split (RMSE 25.97 vs 27.63; MAE 18.45 vs 21.21) despite extensive architecture search and Bayesian optimization. Three technical "
        "considerations are consistent with this outcome. First, daily series length (~1,096 training samples, 1,339 Train+Val) sits within the "
        "small-data regime where parsimonious linear state-space models with fewer than a dozen free parameters are less prone to variance than "
        "multi-layer neural networks with thousands of weights. Second, CPCB daily AQI behaves largely as a locally persistent autoregressive "
        "process anchored linearly to yesterday's PM2.5; first-differencing (d=1) tracks downward annual drift (112.86 -> 103.42 -> 95.73) "
        "naturally without over-anchoring to 2016–2018 mean levels. Third, validation selection across 180+ deep learning configurations introduced "
        "optimistic validation selection bias: while CNN-LSTM beat ARIMAX on Validation (25.63 vs 28.91), part of that advantage may reflect "
        "overfitting to the 243-day Validation regime rather than transferable generalization to the Test regime."
    )

    doc.add_heading("10.2 Festivals as the Primary Failure Mode and Regime-Shift Sensitivity", level=2)
    add_body(
        doc,
        "Every evaluated model relies solely on lagged ambient sensors and smooth seasonal harmonics. Because Diwali pyrotechnics and Bhogi "
        "bonfires inject sharp, single-day emission impulses disconnected from preceding-day PM2.5, all models exhibit large negative errors on "
        "the top-5 worst days (e.g., -140.78 on Diwali 2019). Without explicit calendar event flags, historical sensor lags cannot anticipate "
        "cultural discontinuities. During the 2020 COVID lockdown stress test, a possible reason Pure ARIMA (RMSE 13.51, bias -1.15) and "
        "Persistence (14.45, bias -0.31) remained accurate is that pure univariate differencing adapts immediately to the preceding day's new "
        "low-emission baseline, whereas models relying on pre-lockdown seasonal priors or multivariate weights exhibited larger biases (-8.32 for "
        "ARIMAX; +7.84 and +10.55 for LSTMs)."
    )

    doc.add_heading("10.3 Potential Operational Use", level=2)
    add_body(
        doc,
        "Day-ahead ARIMAX(2,1,1)+F3+PM2.5 forecasts, coupled with split-conformal prediction intervals, can serve as an informational "
        "early-warning input for industrial environmental monitoring. Operational decision thresholds demonstrate measured utility on the test set: "
        "at AQI > 100 (Moderate+), the model achieved 64.5% precision and 76.9% recall (F1 = 0.702 across 52 actual alert days); at AQI > 200 "
        "(Poor+), it achieved 85.7% precision and 60.0% recall (F1 = 0.706 across 10 actual alert days, though rare-event sample size is limited). "
        "Festival days represent a known failure mode where all models exhibit large negative forecast bias; addressing this requires explicit "
        "calendar and event indicator features in future work. Finally, because the underlying CPCB monitoring represents city-level ambient air "
        "quality rather than fence-line perimeter monitoring, these forecasts reflect regional atmospheric conditions and do not isolate the CPCL "
        "Manali refinery's specific operational emissions contribution."
    )

    # =========================================================================
    # SECTION 11: LIMITATIONS & FUTURE SCOPE
    # =========================================================================
    doc.add_heading("11. Limitations & Future Scope", level=1)
    doc.add_heading("11.1 Project Limitations", level=2)
    add_bullet(doc, "City-Level Ambient vs Fence-Line Resolution: ",
               "Station-level gaps at Manali (TN002 depth, TN003 31.28% missingness) necessitated city-level Chennai data, which blends refinery "
               "emissions with urban vehicular, port, and domestic signals.")
    add_bullet(doc, "Absence of Meteorological Predictors: ",
               "Planetary boundary layer height, wind speed and direction, relative humidity, and atmospheric stability fields were absent from "
               "the CPCB dataset, limiting non-festival stagnation prediction.")
    add_bullet(doc, "Severe-Spike Sparsity and Non-Exchangeability: ",
               "Only 10 Test days exceeded AQI 200 (5.5%), restricting statistical confidence for extreme alerts, while distribution drift prevents "
               "time-series observations from strictly satisfying conformal exchangeability assumptions.")

    doc.add_heading("11.2 Future Scope", level=2)
    add_bullet(doc, "On-Site Refinery CEMS and Fence-Line Integration: ",
               "Combine Continuous Emission Monitoring System (CEMS) stack logs from CPCL Manali with dedicated perimeter micro-sensors to isolate "
               "refinery-attributable ambient increments.")
    add_bullet(doc, "Coupling with Numerical Weather Prediction (NWP): ",
               "Incorporate forecasted wind vectors and boundary-layer depth to improve stagnation forecasting on non-festival spike days.")
    add_bullet(doc, "Calendar Event Indicators and Adaptive Updating: ",
               "Add explicit festival binary/lead-time indicators (Diwali, Bhogi) and online rolling recalibration to mitigate structural regime shifts.")

    # =========================================================================
    # SECTION 12: CONCLUSION
    # =========================================================================
    doc.add_heading("12. Conclusion", level=1)
    add_body(
        doc,
        "This study built a reproducible, leakage-free pipeline for forecasting daily Chennai ambient AQI around Indian Oil Corporation's CPCL "
        "Manali refinery zone across classical state-space and deep learning paradigms. Across 182 held-out Test days, ARIMAX(2,1,1) combined "
        "with three annual Fourier harmonics and 1-day lagged PM2.5 emerged as the top day-ahead model (RMSE 25.97, MAE 18.45, R² 0.6657, CPCB "
        "accuracy 76.92%), outperforming a Hybrid ARIMAX+LSTM model (26.81), CNN-LSTM (27.63), Pure ARIMA (30.58), Univariate LSTM (31.73), and "
        "naive persistence (32.10), with exogenous PM2.5 improving accuracy by 12.9%–15.1% across both paradigms."
    )
    add_body(
        doc,
        "Rigorous evaluation also established important empirical boundaries: under Bonferroni correction across 10 Diebold–Mariano tests, only "
        "ARIMAX vs Hybrid remained statistically significant (p = 0.0001); in 7-day-ahead forecasting, validation-selected Epoch-2 Direct LSTM "
        "outperformed persistence only at horizons 5–7 while recursive ARIMA-Fourier led all horizons; and during the 2020 COVID-19 lockdown, "
        "univariate Pure ARIMA (RMSE 13.51) and persistence (14.45) proved most resilient to sudden emission drops. Together with split-conformal "
        "uncertainty intervals (96.70% Test coverage) and clear city-level scope caveats, the ARIMAX framework offers a transparent baseline for "
        "proactive environmental monitoring."
    )

    # =========================================================================
    # REFERENCES (IEEE STYLE — STRICTLY VERIFIABLE ALLOWED SOURCES ONLY)
    # =========================================================================
    doc.add_heading("References", level=1)
    refs = [
        "[1] Vopani (R. Rao), \"Air Quality Data in India (2015–2020),\" Kaggle. [Online]. Available: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india.",
        "[2] Central Pollution Control Board (CPCB), National Air Quality Index, Ministry of Environment, Forest and Climate Change, Government of India, New Delhi, 2014.",
        "[3] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, Time Series Analysis: Forecasting and Control, 5th ed. Hoboken, NJ, USA: Wiley, 2015.",
        "[4] R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. Melbourne, Australia: OTexts. [Online]. Available: https://otexts.com/fpp3/.",
        "[5] S. Hochreiter and J. Schmidhuber, \"Long short-term memory,\" Neural Computation, vol. 9, no. 8, 1997.",
        "[6] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, \"Learning phrase representations using RNN encoder–decoder for statistical machine translation,\" in Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP), 2014.",
        "[7] R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning, \"STL: A seasonal-trend decomposition procedure based on loess,\" Journal of Official Statistics, vol. 6, no. 1, 1990.",
        "[8] D. A. Dickey and W. A. Fuller, \"Distribution of the estimators for autoregressive time series with a unit root,\" Journal of the American Statistical Association, vol. 74, no. 366a, 1979.",
        "[9] D. Kwiatkowski, P. C. B. Phillips, P. Schmidt, and Y. Shin, \"Testing the null hypothesis of stationarity against the alternative of a unit root,\" Journal of Econometrics, vol. 54, 1992.",
        "[10] G. M. Ljung and G. E. P. Box, \"On a measure of lack of fit in time series models,\" Biometrika, vol. 65, no. 2, 1978.",
        "[11] F. X. Diebold and R. S. Mariano, \"Comparing predictive accuracy,\" Journal of Business & Economic Statistics, vol. 13, no. 3, 1995.",
        "[12] D. Harvey, S. Leybourne, and P. Newbold, \"Testing the equality of prediction mean squared errors,\" International Journal of Forecasting, vol. 13, no. 2, 1997.",
        "[13] Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,\" in Proc. Int. Conf. Machine Learning (ICML), 2016.",
        "[14] A. N. Angelopoulos and S. Bates, \"A gentle introduction to conformal prediction and distribution-free uncertainty quantification,\" arXiv:2107.07511, 2021.",
        "[15] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, \"Optuna: A next-generation hyperparameter optimization framework,\" in Proc. ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining (KDD), 2019.",
        "[16] J. Cohen, \"Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit,\" Psychological Bulletin, vol. 70, no. 4, 1968.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        r = p.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10.5)

    # =========================================================================
    # APPENDIX A & APPENDIX B
    # =========================================================================
    doc.add_heading("Appendix A: Environment and Reproducibility", level=1)
    add_body(
        doc,
        "All experiments were executed deterministically on Windows 11 using Python 3.11.0 (64-bit) in a dedicated virtual environment (.venv) "
        "with Jupyter kernel ucs321-aqi. Deep learning models trained on native CPU (TensorFlow 2.16.1) with fixed random seeds [42, 123, 456, 789, 1011]. "
        "Key pinned packages in requirements.txt include numpy==1.26.4, pandas==2.2.2, scipy==1.13.1, scikit-learn==1.5.0, statsmodels==0.14.2, "
        "pmdarima==2.0.4, tensorflow==2.16.1, optuna==3.6.1, matplotlib==3.8.4, and seaborn==0.13.2."
    )
    add_body(doc, "Exact End-to-End Execution Order:", bold_prefix="")
    repro_steps = [
        "1. notebooks/02_data_audit.ipynb — Missingness audit and series selection.",
        "2. notebooks/03_preprocessing.ipynb — TN004 anomaly removal, imputation, upper winsorization, chronological splits, and train-only scaling.",
        "3. notebooks/04_eda.ipynb — Trend, seasonality, STL, correlation, and ADF/KPSS diagnostics.",
        "4. notebooks/05_arima.ipynb — Baselines, 80+ ARIMA/ARIMAX grid search, walk-forward Test/Stress and 175-origin multi-step evaluation.",
        "5. python src/train_dl.py -> python src/retrain_7day.py -> python src/train_univariate_lstm.py -> notebooks/06_lstm.ipynb — Deep learning pipeline.",
        "6. python src/run_phase7.py -> python src/draw_flow_diagram.py -> notebooks/07_evaluation.ipynb -> python src/check_consistency.py — Final benchmark and verification.",
    ]
    for step in repro_steps:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.1
        p.paragraph_format.space_after = Pt(1.5)
        r = p.add_run(step)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10.5)

    doc.add_heading("Appendix B: Top-10 Classical Grid Search & Optuna Best Hyperparameters", level=1)
    p5_grid = pd.read_csv('outputs/results/p5_grid_search.csv').sort_values('val_RMSE').head(10)
    grid_rows = []
    for rank, (_, r) in enumerate(p5_grid.iterrows(), 1):
        grid_rows.append([
            str(rank), str(r['family']), str(r['order']), str(int(r['K'])),
            str(r['has_pm25']), f"{r['AIC']:.2f}", f"{r['BIC']:.2f}",
            f"{r['val_RMSE']:.3f}", f"{r['val_MAE']:.3f}"
        ])
    add_styled_table(
        doc, "B.1",
        "Top 10 ARIMA/ARIMAX Configurations Ranked by Validation RMSE (outputs/results/p5_grid_search.csv)",
        ["Rank", "Model Family", "Order", "K", "PM2.5", "AIC", "BIC", "Val RMSE", "Val MAE"],
        grid_rows,
        col_widths=[0.45, 1.5, 0.7, 0.35, 0.55, 0.85, 0.85, 0.75, 0.75],
        highlight_rows={0}
    )

    add_styled_table(
        doc, "B.2",
        "Optuna Best Hyperparameters (Trial #10, CNN-LSTM, Feature Set CORE, Lookback L=7)",
        ["Hyperparameter", "Selected Value", "Role in Architecture"],
        [
            ["units_l1 / units_l2", "80 / 48", "Conv1D / LSTM hidden filter and state dimensions"],
            ["dropout / recurrent_dropout", "0.15 / 0.20", "Feedforward & recurrent regularization (also used in MC Dropout)"],
            ["learning_rate", "0.00438", "Adam step size (67.9% HPO parameter importance)"],
            ["batch_size", "16", "Mini-batch size (15.3% HPO importance)"],
            ["l2_reg", "1.11e-06", "Weight decay regularization penalty"],
            ["loss_fn", "huber", "Robust loss reducing extreme-spike gradient distortion"],
            ["median_epochs (1-day / 7-day)", "18 / 2", "Validation-selected training epochs for Train+Val retraining"],
        ],
        col_widths=[2.0, 1.4, 3.5],
        highlight_rows={0, 2}
    )

    # Populate List of Figures & List of Tables placeholders
    p_lof_placeholder.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_lof_placeholder.paragraph_format.line_spacing = 1.05
    p_lof_placeholder.paragraph_format.space_after = Pt(2)
    for item in FIGURE_LIST:
        r = p_lof_placeholder.add_run(item + "\n")
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)

    p_lot_placeholder.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_lot_placeholder.paragraph_format.line_spacing = 1.05
    p_lot_placeholder.paragraph_format.space_after = Pt(2)
    for item in TABLE_LIST:
        r = p_lot_placeholder.add_run(item + "\n")
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)

    doc.save(DOCX_PATH)
    print(f"Saved Word report: {DOCX_PATH}")


def export_pdf_via_word():
    """Update Word fields (TOC + Page numbers) and export to PDF via MS Word COM."""
    try:
        import win32com.client
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(DOCX_PATH)
        doc.Fields.Update()
        for toc in doc.TablesOfContents:
            toc.Update()
        doc.Save()
        doc.SaveAs(PDF_PATH, FileFormat=17)  # wdFormatPDF = 17
        doc.Close(False)
        word.Quit()
        print(f"Exported PDF via MS Word COM (with updated TOC): {PDF_PATH}")
        return True
    except Exception as e:
        print(f"Word COM export failed ({e}), trying docx2pdf fallback...")
        try:
            from docx2pdf import convert
            convert(DOCX_PATH, PDF_PATH)
            print(f"Exported PDF via docx2pdf: {PDF_PATH}")
            return True
        except Exception as e2:
            print(f"PDF conversion failed: {e2}")
            return False


if __name__ == '__main__':
    build_report()
    export_pdf_via_word()

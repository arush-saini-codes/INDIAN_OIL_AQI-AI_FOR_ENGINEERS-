"""src/build_ppt.py
Generates the 18-slide widescreen presentation:
  Presentation/UCS321_AQI_Forecasting.pptx
and renders all 18 slides to PNG via PowerPoint COM for visual quality verification.
"""

import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

os.makedirs('Presentation', exist_ok=True)
PPTX_PATH = os.path.abspath('Presentation/UCS321_AQI_Forecasting.pptx')

# Theme Palette
NAVY = RGBColor(0x1B, 0x36, 0x5D)
DARK_TEXT = RGBColor(0x0F, 0x17, 0x2A)
MUTED_TEXT = RGBColor(0x47, 0x55, 0x69)
ACCENT = RGBColor(0x0D, 0x94, 0x88)      # Teal accent
LIGHT_BG = RGBColor(0xF1, 0xF5, 0xF9)
ROW_ALT = RGBColor(0xF8, 0xFA, 0xFC)
ROW_HL = RGBColor(0xE0, 0xF2, 0xFE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER_GRAY = RGBColor(0xCB, 0xD5, 0xE1)

FONT_FAMILY = "Segoe UI"
TOTAL_SLIDES = 18


def add_header_and_footer(slide, title_text, slide_num):
    # Top navy header bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    # Thin teal accent strip under header bar
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.05), Inches(13.333), Inches(0.06))
    strip.fill.solid()
    strip.fill.fore_color.rgb = ACCENT
    strip.line.fill.background()

    # Slide Title (32 pt bold white)
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.16), Inches(12.2), Inches(0.78))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title_text
    r.font.name = FONT_FAMILY
    r.font.size = Pt(32)
    r.font.bold = True
    r.font.color.rgb = WHITE

    # Footer separator & text
    f_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(7.02), Inches(12.23), Inches(0.015))
    f_line.fill.solid()
    f_line.fill.fore_color.rgb = BORDER_GRAY
    f_line.line.fill.background()

    ft_left = slide.shapes.add_textbox(Inches(0.55), Inches(7.08), Inches(9.0), Inches(0.32))
    p_l = ft_left.text_frame.paragraphs[0]
    r_l = p_l.add_run()
    r_l.text = "UCS321 – EST Project  |  IndianOil (CPCL Manali) AQI Forecasting"
    r_l.font.name = FONT_FAMILY
    r_l.font.size = Pt(12)
    r_l.font.color.rgb = MUTED_TEXT

    ft_right = slide.shapes.add_textbox(Inches(11.0), Inches(7.08), Inches(1.78), Inches(0.32))
    p_r = ft_right.text_frame.paragraphs[0]
    p_r.alignment = PP_ALIGN.RIGHT
    r_r = p_r.add_run()
    r_r.text = f"Slide {slide_num} / {TOTAL_SLIDES}"
    r_r.font.name = FONT_FAMILY
    r_r.font.size = Pt(12)
    r_r.font.bold = True
    r_r.font.color.rgb = NAVY


def add_bullets_box(slide, left, top, width, height, bullets, font_size=20, space_after=14):
    """Add formatted bullets (max 5 bullets recommended, >=20 pt body text)."""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    for idx, b in enumerate(bullets):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after)
        p.line_spacing = 1.18

        # Bullet marker
        rm = p.add_run()
        rm.text = "▪  "
        rm.font.name = FONT_FAMILY
        rm.font.size = Pt(font_size)
        rm.font.bold = True
        rm.font.color.rgb = ACCENT

        if isinstance(b, tuple):
            lead, rest = b
            rl = p.add_run()
            rl.text = lead + " "
            rl.font.name = FONT_FAMILY
            rl.font.size = Pt(font_size)
            rl.font.bold = True
            rl.font.color.rgb = NAVY

            rr = p.add_run()
            rr.text = rest
            rr.font.name = FONT_FAMILY
            rr.font.size = Pt(font_size)
            rr.font.color.rgb = DARK_TEXT
        else:
            rr = p.add_run()
            rr.text = b
            rr.font.name = FONT_FAMILY
            rr.font.size = Pt(font_size)
            rr.font.color.rgb = DARK_TEXT


def add_picture_fitted(slide, img_path, box_left, box_top, box_width, box_height):
    """Insert image centered inside bounding box while preserving natural aspect ratio."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    img_ratio = iw / ih
    box_ratio = box_width / box_height
    if img_ratio > box_ratio:
        w = box_width
        h = box_width / img_ratio
    else:
        h = box_height
        w = box_height * img_ratio
    x = box_left + (box_width - w) / 2.0
    y = box_top + (box_height - h) / 2.0
    slide.shapes.add_picture(img_path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def set_notes(slide, text):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = text.strip()


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    accent_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.14), Inches(5.1))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = ACCENT
    accent_bar.line.fill.background()

    tb_inst = s1.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(11.3), Inches(0.6))
    p = tb_inst.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "THAPAR INSTITUTE OF ENGINEERING & TECHNOLOGY  |  UCS321 – EST PROJECT"
    r.font.name = FONT_FAMILY
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x99, 0xF6, 0xE4)

    tb_title = s1.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(11.3), Inches(2.2))
    tf = tb_title.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.line_spacing = 1.2
    r = p.add_run()
    r.text = "Forecasting Air Quality Index around IndianOil Refinery Zones using ARIMA and LSTM Models"
    r.font.name = FONT_FAMILY
    r.font.size = Pt(36)
    r.font.bold = True
    r.font.color.rgb = WHITE

    tb_sub = s1.shapes.add_textbox(Inches(1.2), Inches(4.2), Inches(11.3), Inches(0.6))
    p = tb_sub.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "Chennai Corridor Study (CPCL Manali Refinery Zone, 2016–2020)"
    r.font.name = FONT_FAMILY
    r.font.size = Pt(22)
    r.font.italic = True
    r.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)

    tb_meta = s1.shapes.add_textbox(Inches(1.2), Inches(5.15), Inches(11.3), Inches(1.5))
    tf = tb_meta.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.line_spacing = 1.4
    r = p.add_run()
    r.text = "Group Members: [Group Member Names & Roll Numbers]\nInstructor: [Instructor Name]   |   Date: [Date]"
    r.font.name = FONT_FAMILY
    r.font.size = Pt(20)
    r.font.color.rgb = WHITE

    set_notes(s1, (
        "Good morning everyone. Today our team presents our UCS321 Engineering Design project on forecasting the daily "
        "Air Quality Index around Indian Oil Corporation refinery zones, focusing on the CPCL Manali refinery corridor "
        "in Chennai. In this presentation, we compare classical ARIMA and ARIMAX time-series models against deep learning "
        "LSTM architectures across 1-day and 7-day forecast horizons, as well as a real-world lockdown stress test. Let us begin "
        "with the problem statement and our main engineering objectives."
    ))

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & OBJECTIVES
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s2, "Problem Statement & Objectives", 2)

    q_box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(1.45), Inches(12.0), Inches(1.75))
    q_box.adjustments[0] = 0.06
    q_box.fill.solid()
    q_box.fill.fore_color.rgb = LIGHT_BG
    q_box.line.color.rgb = ACCENT
    q_box.line.width = Pt(2)
    tf_q = q_box.text_frame
    tf_q.word_wrap = True
    tf_q.margin_left = Inches(0.35)
    tf_q.margin_right = Inches(0.35)
    tf_q.margin_top = Inches(0.2)
    p = tf_q.paragraphs[0]
    p.line_spacing = 1.25
    r = p.add_run()
    r.text = (
        "\"Indian Oil Corporation seeks to forecast the Air Quality Index (AQI) around its refinery zones. "
        "Using historical pollutant levels, an ARIMA and LSTM-based forecasting model is needed to predict "
        "AQI values and support proactive emission control.\""
    )
    r.font.name = FONT_FAMILY
    r.font.size = Pt(21)
    r.font.italic = True
    r.font.color.rgb = NAVY

    add_bullets_box(s2, 0.75, 3.55, 11.8, 3.2, [
        ("Daily Horizons:", "Forecast ambient AQI 1 day ahead and 7 days ahead."),
        ("Dual Paradigms:", "Benchmark classical ARIMA/ARIMAX against tuned LSTM and hybrid models."),
        ("Strict Leakage Control:", "Chronological Train, Validation, Test, and COVID Stress splits."),
        ("Uncertainty & Alerts:", "Provide calibrated prediction intervals and CPCB threshold alert metrics."),
    ], font_size=22, space_after=18)

    set_notes(s2, (
        "Our problem statement comes directly from the Indian Oil Corporation brief: we need to predict AQI around refinery "
        "zones using historical pollutant levels so environmental teams can get early warnings. To solve this cleanly, we set "
        "four objectives. First, predict both 1-day-ahead and 7-day-ahead daily AQI. Second, build and compare classical ARIMA "
        "family models against deep learning LSTM variants. Third, enforce strict chronological splitting with zero data leakage. "
        "And fourth, evaluate alert thresholds, statistical significance, and prediction uncertainty."
    ))

    # =========================================================================
    # SLIDE 3: WHY CHENNAI / CPCL MANALI + DATASET OVERVIEW
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s3, "Location Choice & Dataset Overview", 3)

    add_bullets_box(s3, 0.65, 1.45, 5.7, 5.2, [
        ("CPCL Manali:", "10.5 MMTPA IndianOil subsidiary refinery in north Chennai."),
        ("Daily Granularity:", "Matches 24-hour CPCB standards; filters hourly noise."),
        ("City-Level Selection:", "Manali TN002 (256d) and TN003 (31.3% missing) failed quality rules."),
        ("2016–2020 Series:", "1,643 continuous days; 2015 excluded for 82-day outage."),
        ("Four Features Sets:", "CORE (PM2.5, CO, O3, NO2, SO2) and ALL9."),
    ], font_size=20, space_after=16)

    # Clean Split Overview Table on Right
    t_shape = s3.shapes.add_table(6, 4, Inches(6.6), Inches(1.55), Inches(6.15), Inches(4.5))
    tbl = t_shape.table
    col_w = [1.55, 2.1, 0.9, 1.6]
    for idx, w in enumerate(col_w):
        tbl.columns[idx].width = Inches(w)
    s3_rows = [
        ["Split", "Date Window", "Days", "Mean AQI"],
        ["Train", "2016-01-01 to 2018-12-31", "1,096", "112.86"],
        ["Validation", "2019-01-01 to 2019-08-31", "243", "103.42"],
        ["Test", "2019-09-01 to 2020-02-29", "182", "95.73"],
        ["Stress", "2020-03-01 to 2020-06-30", "122", "78.81"],
        ["Full Series", "2016-01-01 to 2020-06-30", "1,643", "107.04"],
    ]
    for r_i, row_vals in enumerate(s3_rows):
        for c_i, val in enumerate(row_vals):
            cell = tbl.cell(r_i, c_i)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r_i == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r_i in (3, 4):
                cell.fill.fore_color.rgb = ROW_HL
            else:
                cell.fill.fore_color.rgb = ROW_ALT if r_i % 2 == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c_i < 2 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = val
            run.font.name = FONT_FAMILY
            run.font.size = Pt(16)
            run.font.bold = (r_i == 0 or r_i in (3, 4))
            run.font.color.rgb = WHITE if r_i == 0 else DARK_TEXT

    set_notes(s3, (
        "We chose Chennai because Indian Oil operates the CPCL Manali refinery there. Using the Kaggle CPCB daily dataset, "
        "we applied a formal data-quality rule: Manali station TN002 only had 256 days of history, and Manali station TN003 "
        "was missing 31.28 percent of its data with a 411-day gap. So following our rule, we selected the Chennai city-level "
        "daily series starting January 2016, excluding 2015 due to an 82-day monitoring outage. As you see in the table, we "
        "split the 1,643 days chronologically into Train, Validation, Test, and an isolated 2020 COVID lockdown stress window."
    ))

    # =========================================================================
    # SLIDE 4: METHODOLOGY FLOW DIAGRAM
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s4, "End-to-End Methodology Pipeline", 4)

    add_picture_fitted(s4, "outputs/figures/flow_diagram_compact.png", 0.55, 1.25, 6.2, 5.65)
    add_bullets_box(s4, 7.05, 1.65, 5.7, 4.9, [
        ("Leakage-Free Prep:", "Imputers, upper clipping, and scalers fit on Train only."),
        ("Diagnostics:", "STL, Spearman correlation, ADF/KPSS, and ACF/PACF."),
        ("Parallel Modeling:", "Classical ARIMA/ARIMAX alongside staged LSTM and Hybrid."),
        ("Fair Evaluation:", "Diebold–Mariano tests, alerts, intervals, and stress testing."),
    ], font_size=21, space_after=22)

    set_notes(s4, (
        "This flow diagram summarizes our full six-stage engineering workflow. After acquiring and auditing the Chennai daily "
        "series, we run a strictly ordered preprocessing pipeline where all scalers and imputation statistics are fitted only on "
        "the 2016 to 2018 training split. Next, exploratory analysis guides model orders and feature selection. We then run two "
        "parallel modeling tracks: classical ARIMA and ARIMAX on the left, and staged deep learning LSTMs plus a residual hybrid "
        "on the right. Finally, every model meets on identical Test and lockdown Stress benchmarks."
    ))

    # =========================================================================
    # SLIDE 5: PREPROCESSING PIPELINE & SENSOR ANOMALY STORY
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s5, "Preprocessing & Sensor Anomaly Removal", 5)

    add_bullets_box(s5, 0.6, 1.45, 5.4, 5.3, [
        ("TN004 CO Fault:", "15 anomaly days in Aug–Sep 2016 flagged and re-imputed."),
        ("Target Max Fixed:", "Peak drops from faulty 449 to Diwali 2017 (431)."),
        ("Gap Imputation:", "<=3d linear fill; 4–30d DOY train median; 0% missing after."),
        ("Upper Clipping:", "Pollutants clipped at 99.5th pct; AQI spikes preserved."),
    ], font_size=20, space_after=18)
    add_picture_fitted(s5, "outputs/figures/p3_final_cleaned_aqi_splits.png", 6.15, 1.4, 6.65, 5.35)

    set_notes(s5, (
        "During data inspection, we discovered a major sensor fault at station TN004 in August and September 2016, where CO "
        "readings jumped wildly and pushed city AQI up to 449 while PM2.5 stayed normal and station TN001 read 64. We confirmed "
        "15 faulty days against station records, set them to missing, and re-imputed them using our gap-adaptive method. That "
        "brought the real peak to Diwali 2017 at 431. We also clipped pollutant inputs at the upper 99.5th percentile while leaving "
        "target AQI spikes untouched so real pollution episodes remain intact."
    ))

    # =========================================================================
    # SLIDE 6: EDA 1 - TREND & SEASONALITY
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s6, "EDA 1: Trend, Seasonality & Key Drivers", 6)

    add_bullets_box(s6, 0.6, 1.45, 5.2, 5.2, [
        ("Downward Drift:", "Mean AQI fell from 112.86 (Train) to 95.73 (Test)."),
        ("Annual Cycle:", "Winter (Nov–Jan) peaks highest; monsoon (Jun–Aug) lowest."),
        ("No Weekly Cycle:", "Day-of-week averages stay flat (~3 AQI variation)."),
        ("Primary Driver:", "PM2.5 correlates most strongly with next-day AQI (r = 0.606)."),
    ], font_size=20, space_after=18)
    add_picture_fitted(s6, "outputs/figures/p4_seasonality_analysis.png", 5.95, 1.35, 6.85, 5.45)

    set_notes(s6, (
        "Looking at exploratory analysis, seasonal boxplots reveal two key patterns on the training data. First, there is a clear "
        "multi-year downward trend in AQI from 112.86 in Train down to 95.73 in Test. Second, annual seasonality is strong, driven "
        "by winter inversion peaks from November to January and monsoon washout lows from June to August, while weekly day-of-week "
        "averages remain essentially flat. Over 93 percent of days fall in the Satisfactory or Moderate CPCB bands. Finally, PM2.5 "
        "is the strongest chemical predictor for next-day AQI, with a lag-1 correlation of 0.606."
    ))

    # =========================================================================
    # SLIDE 7: EDA 2 - STATIONARITY & ACF/PACF (d=0 vs d=1)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s7, "EDA 2: Stationarity & Differencing (d=0 vs d=1)", 7)

    add_bullets_box(s7, 0.6, 1.45, 5.2, 5.2, [
        ("Conflicting Raw Tests:", "Raw ADF p = 1.17e-6 vs KPSS p = 0.01."),
        ("Differencing (d=1):", "Both agree stationary: ADF p = 5.77e-24, KPSS p = 0.10."),
        ("PACF Lag-1 Cutoff:", "Sharp lag-1 PACF (0.697) guides low AR order."),
        ("d=1 Beats d=0:", "Validation shows d=1 (RMSE 28.91 vs 30.13) tracks drift."),
    ], font_size=20, space_after=18)
    add_picture_fitted(s7, "outputs/figures/p4_acf_pacf.png", 5.95, 1.35, 6.85, 5.45)

    set_notes(s7, (
        "Stationarity testing gave us a textbook empirical nuance. On raw AQI, the ADF test rejected a unit root, but KPSS rejected "
        "stationarity due to the gradual multi-year trend. Once we applied first differencing with d equals 1, both ADF and KPSS "
        "agreed the series was stationary. Notice the negative lag-1 ACF after differencing, which can signal over-differencing, so "
        "we tested both d equals 0 and d equals 1 fairly on the Validation set. Ultimately d equals 1 won because it adapts better "
        "to multi-day local level drift."
    ))

    # =========================================================================
    # SLIDE 8: ARIMA/ARIMAX APPROACH + FINAL MODEL
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s8, "Classical Track: ARIMA & ARIMAX Modelling", 8)

    add_bullets_box(s8, 0.6, 1.45, 5.3, 5.2, [
        ("Three Baselines:", "Persistence (32.10), 7-Day MA (42.44), Seasonal 365d (71.61)."),
        ("80+ Grid Search:", "Evaluated ARIMA, +Fourier (K=1..3), and +PM2.5(t-1)."),
        ("Champion Model:", "ARIMAX(2,1,1) + Fourier(K=3) + PM2.5 (Val RMSE 28.91)."),
        ("Clean Diagnostics:", "Ljung–Box p = 0.607 at lag 20; unbiased white-noise core."),
    ], font_size=20, space_after=18)
    add_picture_fitted(s8, "outputs/figures/p5_model_diagnostics.png", 6.05, 1.35, 6.75, 5.45)

    set_notes(s8, (
        "In the classical modeling track, we replaced computationally heavy 365-day seasonal differencing with K equals 3 pairs of "
        "annual Fourier sine and cosine terms. We searched over 80 configurations spanning Pure ARIMA, ARIMA with Fourier harmonics, "
        "and ARIMAX adding lagged PM2.5. Note that AIC cannot be compared across d equals 0 and d equals 1, so we ranked candidates "
        "strictly by Validation RMSE. Our champion, ARIMAX(2,1,1) with 3 Fourier pairs and lagged PM2.5, reached 28.91 Validation "
        "RMSE and passed Ljung–Box white-noise diagnostics."
    ))

    # =========================================================================
    # SLIDE 9: LSTM APPROACH - STAGED SEARCH
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s9, "Deep Learning Track: Staged LSTM Pipeline", 9)

    add_bullets_box(s9, 0.6, 1.45, 5.4, 5.2, [
        ("Stage 1 (80 runs):", "Lookback L=7 with CORE features won (Val RMSE 27.88)."),
        ("Stage 2 (40 runs):", "CNN-LSTM beat GRU, BiLSTM, Stacked LSTM (26.24)."),
        ("Stage 3 (60 trials):", "Optuna TPE tuned units (80,48), Huber loss (Val 25.50)."),
        ("5-Seed Ensembles:", "Retrained on Train+Val; plus Hybrid & Direct 7-Day."),
    ], font_size=20, space_after=18)
    add_picture_fitted(s9, "outputs/figures/p6_stage2_architecture_comparison.png", 6.15, 1.4, 6.65, 5.35)

    set_notes(s9, (
        "For deep learning, instead of guessing one LSTM architecture, we ran a structured three-stage search with 5 random seeds "
        "each. First, an 80-run grid crossed four feature sets with four lookbacks, showing that a short 7-day window with CORE "
        "pollutants performed best at 27.88 Validation RMSE. Second, our 40-run tournament crowned a 1-D CNN-LSTM architecture at "
        "26.24 RMSE. Third, 60 Optuna Bayesian trials tuned hyperparameters with Huber loss, reaching 25.50 on Validation before "
        "retraining 5-seed ensembles on Train plus Validation."
    ))

    # =========================================================================
    # SLIDE 10: MASTER RESULTS TABLE (TEST, 1-DAY)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s10, "Master Benchmark Results (Test Split, 1-Day Ahead)", 10)

    t10_shape = s10.shapes.add_table(9, 8, Inches(0.55), Inches(1.45), Inches(12.23), Inches(5.15))
    t10 = t10_shape.table
    w10 = [1.80, 3.05, 1.10, 1.10, 1.25, 1.20, 1.43, 1.30]
    for i, w in enumerate(w10):
        t10.columns[i].width = Inches(w)

    rows10 = [
        ["Model Family", "Specific Model", "RMSE", "MAE", "MAPE (%)", "R²", "CPCB Acc", "QWK"],
        ["Classical Champion", "ARIMAX(2,1,1)+F3+PM2.5", "25.97", "18.45", "20.25", "0.6657", "76.92%", "0.6471"],
        ["Ensemble Hybrid", "Hybrid ARIMAX + LSTM", "26.81", "19.91", "22.58", "0.6435", "74.73%", "0.6296"],
        ["DL Champion", "CNN-LSTM (L=7, CORE)", "27.63", "21.21", "23.92", "0.6214", "68.68%", "0.5384"],
        ["Classical", "Pure ARIMA(2,1,1)", "30.58", "19.23", "19.33", "0.5362", "74.18%", "0.5587"],
        ["Deep Learning", "Univariate LSTM (L=7)", "31.73", "22.87", "25.43", "0.5008", "70.88%", "0.5226"],
        ["Baseline", "Naive Persistence (182d)", "32.10", "19.56", "19.40", "0.4889", "73.63%", "0.5951"],
        ["Baseline", "7-Day Moving Average", "42.44", "26.49", "27.19", "0.1067", "64.84%", "0.3405"],
        ["Baseline", "Seasonal Naive (365d)", "71.61", "53.86", "61.80", "-1.5426", "36.81%", "-0.0554"],
    ]
    for r_i, rv in enumerate(rows10):
        for c_i, val in enumerate(rv):
            cell = t10.cell(r_i, c_i)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r_i == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r_i == 1:
                cell.fill.fore_color.rgb = ROW_HL
            else:
                cell.fill.fore_color.rgb = ROW_ALT if r_i % 2 == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c_i < 2 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = val
            run.font.name = FONT_FAMILY
            run.font.size = Pt(16)
            run.font.bold = (r_i == 0 or r_i == 1)
            run.font.color.rgb = WHITE if r_i == 0 else (NAVY if r_i == 1 else DARK_TEXT)

    set_notes(s10, (
        "Here are our master 1-day-ahead results on the 182-day Test set. Classical ARIMAX(2,1,1) with Fourier terms and lagged "
        "PM2.5 achieved the strongest overall performance: Test RMSE 25.97, MAE 18.45, R-squared 0.6657, and 76.92 percent CPCB "
        "category accuracy. Notice that the Hybrid ARIMAX plus residual LSTM came second at 26.81 RMSE, followed by CNN-LSTM at "
        "27.63. Under a conservative Bonferroni correction across 10 Diebold–Mariano tests, only the ARIMAX versus Hybrid difference "
        "remains statistically significant at alpha equals 0.005."
    ))

    # =========================================================================
    # SLIDE 11: TEST OVERLAY FIGURE
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s11, "Test Trajectory Overlay & Festival Spike Audit", 11)

    add_picture_fitted(s11, "outputs/figures/p7_test_overlay_zoomed.png", 0.55, 1.3, 8.0, 5.5)
    add_bullets_box(s11, 8.75, 1.65, 4.05, 4.9, [
        ("Close Baseline Tracking:", "Normal days (<148 AQI) tracked with 18.78 ARIMAX RMSE."),
        ("Festival Failure Mode:", "2 of top 5 worst days: Diwali (-140.8), Bhogi (-92.2)."),
        ("Non-Matched Spikes:", "Remaining 3 worst days reflect unmodeled stagnation."),
    ], font_size=20, space_after=20)

    set_notes(s11, (
        "This overlay shows daily predictions on the Test window with a zoomed inset on Diwali 2019. On normal days below 148 AQI, "
        "ARIMAX tracks observations closely with 18.78 RMSE. However, every model under-predicts sudden spikes because preceding-day "
        "sensors cannot see unannounced events. Auditing the 5 worst-forecast days, exactly 2 match verified cultural festivals: "
        "Diwali on October 27 where AQI hit 306 and models missed by over 140 points, and Bhogi on January 14. The remaining 3 "
        "worst days have no calendar match."
    ))

    # =========================================================================
    # SLIDE 12: UNIVARIATE VS MULTIVARIATE 2x2
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s12, "Paradigm vs Information Scope (2x2 Grid)", 12)

    add_picture_fitted(s12, "outputs/figures/p7_2x2_univariate_multivariate.png", 0.65, 1.35, 6.4, 5.45)
    add_bullets_box(s12, 7.25, 1.7, 5.5, 4.8, [
        ("Classical Gain:", "Adding PM2.5 cuts Test RMSE 15.1% (30.58 to 25.97)."),
        ("DL Gain:", "Adding CORE features cuts RMSE 12.9% (31.73 to 27.63)."),
        ("Consistent Lead:", "Classical outperforms DL across both univariate and multivariate."),
        ("Core Takeaway:", "Pollutant inputs matter more than network complexity here."),
    ], font_size=20, space_after=20)

    set_notes(s12, (
        "Our 2-by-2 factorial comparison separates model family from feature information. Adding PM2.5 to classical ARIMA cuts "
        "Test RMSE by 15.1 percent, from 30.58 down to 25.97. Similarly, adding our 5 CORE pollutants to univariate LSTM cuts RMSE "
        "by 12.9 percent, from 31.73 down to 27.63. Across both information scopes, the parsimonious classical models outperform "
        "deep learning, suggesting that on 1,096 training days, exogenous chemical signals drive larger gains than added neural complexity."
    ))

    # =========================================================================
    # SLIDE 13: MULTI-STEP 1-7 DAY RESULTS
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s13, "Multi-Horizon 1–7 Day Forecasting (175 Origins)", 13)

    add_picture_fitted(s13, "outputs/figures/p7_rmse_vs_horizon_final.png", 0.55, 1.35, 7.35, 5.45)
    add_bullets_box(s13, 8.1, 1.6, 4.7, 5.1, [
        ("Leakage-Free Epochs:", "7-day LSTM stopped at Epoch 2 (min Val RMSE 33.88)."),
        ("Beats Persist at h=5–7:", "LSTM wins h=5 (56.08), h=6 (54.48), h=7 (53.04)."),
        ("Loses at h=1–4:", "LSTM (40.39 at h=1) trails Persistence (32.72) early on."),
        ("ARIMA-Fourier Leads:", "Lowest RMSE (32.25 to 50.35) across all horizons."),
    ], font_size=20, space_after=18)

    set_notes(s13, (
        "For 7-day-ahead forecasting across 175 aligned Test origins, we selected the Direct Multi-Step LSTM training epochs strictly "
        "on the Validation set without touching Test data. Validation error across all 7 horizons reached its minimum at Epoch 2, "
        "reflecting the 2016–18 versus 2019 level shift. Retrained for 2 epochs, the 7-day LSTM beats persistence only at horizons "
        "5, 6, and 7, while persistence and recursive ARIMA-Fourier lead horizons 1 through 4. Overall, ARIMA-Fourier achieves the "
        "lowest multi-day RMSE, consistent with Fourier terms constraining dispersion."
    ))

    # =========================================================================
    # SLIDE 14: LOCKDOWN STRESS TEST
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s14, "COVID-19 Lockdown Stress Test (Mean AQI = 78.81)", 14)

    add_picture_fitted(s14, "outputs/figures/p7_stress_overlay.png", 0.55, 1.35, 7.35, 5.45)
    add_bullets_box(s14, 8.1, 1.55, 4.7, 5.2, [
        ("Pure ARIMA Best:", "Lowest stress RMSE (13.51, MAE 9.06, bias -1.15)."),
        ("Persistence Second:", "RMSE 14.45, MAE 9.41, near-zero bias (-0.31)."),
        ("Complex Models Lagged:", "ARIMAX (17.74) and CNN-LSTM (20.28) performed worst."),
        ("Opposite Biases:", "LSTMs: +7.84 / +10.55 bias; ARIMAX bias: -8.32."),
    ], font_size=20, space_after=18)

    set_notes(s14, (
        "Evaluating without retraining on the 122-day COVID lockdown window, where mean AQI dropped to 78.81, revealed sharp "
        "differences in regime resilience. Pure ARIMA achieved the lowest stress RMSE at 13.51, followed closely by naive persistence "
        "at 14.45. By contrast, models using exogenous pollutants or seasonal terms degraded: ARIMAX reached 17.74 RMSE with negative "
        "mean bias of minus 8.32, while Univariate and Multivariate CNN-LSTM reached 15.98 and 20.28 RMSE with positive over-prediction "
        "biases of plus 7.84 and plus 10.55."
    ))

    # =========================================================================
    # SLIDE 15: ALERTS & PREDICTION INTERVALS
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s15, "Operational Alerts & Uncertainty Quantification", 15)

    add_picture_fitted(s15, "outputs/figures/p7_alert_precision_recall.png", 0.5, 1.35, 6.05, 3.6)
    add_picture_fitted(s15, "outputs/figures/p7_interval_comparison.png", 6.75, 1.35, 6.05, 3.6)
    add_bullets_box(s15, 0.65, 5.15, 12.0, 1.7, [
        ("Alerts (ARIMAX):", "AQI >100 F1 = 0.702 (52 days); AQI >200 F1 = 0.706 (10 rare days)."),
        ("Conformal vs MC:", "Split-conformal covered 96.7% (caveat: non-exchangeable drift); MC covered 37.9%."),
    ], font_size=20, space_after=10)

    set_notes(s15, (
        "On the left, we evaluate operational alert thresholds. At AQI above 100, ARIMAX reaches 64.5 percent precision and 76.9 "
        "percent recall for an F1 of 0.702. At AQI above 200, precision rises to 85.7 percent, though only 10 such days occurred in "
        "Test. On the right, split-conformal intervals calibrated on Validation errors achieved 96.70 percent Test coverage for ARIMAX "
        "and 96.15 percent for LSTM, whereas MC-Dropout severely undercovered at 37.91 percent. Note that time-series drift means "
        "exchangeability is not strictly satisfied."
    ))

    # =========================================================================
    # SLIDE 16: KEY FINDINGS
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s16, "Key Project Findings", 16)

    add_bullets_box(s16, 0.85, 1.55, 11.6, 5.2, [
        ("Day-Ahead Champion:", "ARIMAX(2,1,1)+F3+PM2.5 achieved top Test RMSE (25.97) and MAE (18.45)."),
        ("Exogenous PM2.5 Crucial:", "Pollutant inputs lowered error 15.1% (Classical) and 12.9% (DL)."),
        ("Bonferroni Caution:", "Only ARIMAX vs Hybrid (p=0.0001) stayed significant after 10-test adjustment."),
        ("Multi-Step & Lockdown:", "Epoch-2 LSTM beats persistence only at h=5–7; Pure ARIMA leads lockdown (13.51)."),
        ("Potential Operational Use:", "Informational city-level early warning; fence-line monitoring needed to isolate refinery emissions."),
    ], font_size=21, space_after=20)

    set_notes(s16, (
        "To summarize our five core findings: first, ARIMAX(2,1,1) with Fourier harmonics and lagged PM2.5 outperformed all deep "
        "learning and hybrid competitors on 1-day Test RMSE and MAE. Second, exogenous PM2.5 drove 13 to 15 percent error reductions "
        "across both paradigms. Third, Bonferroni correction reminds us that differences among top models are modest. Fourth, Pure "
        "ARIMA proved most resilient during the COVID structural break. Finally, this model provides valuable city-level ambient early "
        "warning, though isolating refinery contributions requires perimeter sensors."
    ))

    # =========================================================================
    # SLIDE 17: LIMITATIONS & FUTURE SCOPE
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s17, "Limitations & Future Scope", 17)

    # Left Card: Limitations
    card_l = s17.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.45), Inches(5.85), Inches(5.25))
    card_l.adjustments[0] = 0.04
    card_l.fill.solid()
    card_l.fill.fore_color.rgb = LIGHT_BG
    card_l.line.color.rgb = NAVY
    card_l.line.width = Pt(1.5)
    tl_hdr = s17.shapes.add_textbox(Inches(0.85), Inches(1.65), Inches(5.35), Inches(0.5))
    p = tl_hdr.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "Key Limitations"
    r.font.name = FONT_FAMILY
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = NAVY
    add_bullets_box(s17, 0.85, 2.3, 5.35, 4.1, [
        ("City-Level Scope:", "Does not isolate CPCL Manali fence-line emissions."),
        ("No Meteorology:", "Wind speed, humidity, and mixing height unavailable."),
        ("Rare Severe Days:", "Only 10 Test days >200 AQI; unmodeled festival spikes."),
    ], font_size=20, space_after=22)

    # Right Card: Future Scope
    card_r = s17.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.88), Inches(1.45), Inches(5.85), Inches(5.25))
    card_r.adjustments[0] = 0.04
    card_r.fill.solid()
    card_r.fill.fore_color.rgb = ROW_HL
    card_r.line.color.rgb = ACCENT
    card_r.line.width = Pt(1.5)
    tr_hdr = s17.shapes.add_textbox(Inches(7.13), Inches(1.65), Inches(5.35), Inches(0.5))
    p = tr_hdr.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "Future Scope"
    r.font.name = FONT_FAMILY
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = ACCENT
    add_bullets_box(s17, 7.13, 2.3, 5.35, 4.1, [
        ("Perimeter CEMS:", "Integrate stack exhausts and refinery fence-line monitors."),
        ("Weather Coupling:", "Add NWP wind vectors and boundary-layer forecasts."),
        ("Event Indicators:", "Encode Diwali/Bhogi flags and rolling model recalibration."),
    ], font_size=20, space_after=22)

    set_notes(s17, (
        "Looking honestly at limitations, our series reflects Chennai city-wide CPCB ambient stations rather than dedicated CPCL "
        "Manali fence-line sensors, and lacks meteorological fields like wind velocity or mixing height. Also, only 10 days exceeded "
        "200 AQI in Test, and sudden festival events remain unpredictable without calendar indicators. In future work, coupling on-site "
        "CEMS stack telemetry with Numerical Weather Prediction forecasts, explicit Diwali and Bhogi binary flags, and adaptive rolling "
        "updates would directly close these gaps."
    ))

    # =========================================================================
    # SLIDE 18: THANK YOU / Q&A
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    bg18 = s18.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg18.fill.solid()
    bg18.fill.fore_color.rgb = NAVY
    bg18.line.fill.background()

    tb_ty = s18.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.333), Inches(1.3))
    p = tb_ty.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Thank You — Questions & Discussion"
    r.font.name = FONT_FAMILY
    r.font.size = Pt(40)
    r.font.bold = True
    r.font.color.rgb = WHITE

    tb_det = s18.shapes.add_textbox(Inches(1.5), Inches(3.8), Inches(10.333), Inches(1.8))
    tf = tb_det.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.line_spacing = 1.45
    r = p.add_run()
    r.text = (
        "UCS321 – EST Project  |  Thapar Institute of Engineering & Technology\n"
        "Champion Model: ARIMAX(2,1,1) + Fourier(K=3) + PM2.5  |  Test RMSE: 25.97\n"
        "[Group Member Names & Roll Numbers]"
    )
    r.font.name = FONT_FAMILY
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor(0x99, 0xF6, 0xE4)

    set_notes(s18, (
        "Thank you for your attention. All code, executed notebooks, frozen benchmark tables, and technical documentation are "
        "reproducibly archived in the project repository. We welcome your questions on our data pipeline, model selection, "
        "statistical testing, or lockdown stress evaluation."
    ))

    prs.save(PPTX_PATH)
    print(f"Saved PowerPoint deck: {PPTX_PATH} ({len(prs.slides)} slides)")


def render_slides_to_png(out_dir=None):
    """Export every slide to PNG using PowerPoint COM automation for visual verification."""
    import tempfile
    if out_dir is None:
        out_dir = os.path.join(tempfile.gettempdir(), 'ucs321_ppt_slides')
    os.makedirs(out_dir, exist_ok=True)
    try:
        import win32com.client
        ppt = win32com.client.Dispatch("PowerPoint.Application")
        pres = ppt.Presentations.Open(PPTX_PATH, WithWindow=False)
        pres.Export(out_dir, "PNG", 1280, 720)
        pres.Close()
        ppt.Quit()
        pngs = sorted([f for f in os.listdir(out_dir) if f.lower().endswith('.png')])
        print(f"Exported {len(pngs)} slide PNGs to {out_dir}")
        return out_dir, pngs
    except Exception as e:
        print(f"PowerPoint COM export failed: {e}")
        return out_dir, []


if __name__ == '__main__':
    build_deck()
    render_slides_to_png()

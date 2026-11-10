"""src/draw_flow_diagram.py
Generates publication-quality pipeline flow diagrams at 300 DPI:
  1. outputs/figures/flow_diagram.png (detailed full pipeline)
  2. outputs/figures/flow_diagram_compact.png (compact stage-level for PPT)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIGURES_DIR = 'outputs/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def draw_box(ax, x, y, w, h, title, subtitle=None, facecolor='#FFFFFF',
             edgecolor='#333333', lw=1.4, title_size=8.5, sub_size=7.0,
             title_color='#111111', sub_color='#333333', radius=0.6, zorder=3):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.15,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    cx = x + w / 2.0
    if subtitle:
        n_lines = subtitle.count('\n') + 1
        if n_lines >= 3:
            cy_title = y + h * 0.74
            cy_sub = y + h * 0.33
        else:
            cy_title = y + h * 0.65
            cy_sub = y + h * 0.30
        ax.text(cx, cy_title, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=title_color,
                family='sans-serif', zorder=zorder + 1)
        ax.text(cx, cy_sub, subtitle, ha='center', va='center',
                fontsize=sub_size, color=sub_color, family='sans-serif',
                linespacing=1.22, zorder=zorder + 1)
    else:
        ax.text(cx, y + h / 2.0, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=title_color,
                family='sans-serif', zorder=zorder + 1)


def draw_container(ax, x, y, w, h, label, facecolor='#F8FAFC', edgecolor='#64748B',
                   label_bg='#64748B', label_size=9.5, radius=1.0, zorder=1):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.2,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.8, zorder=zorder
    )
    ax.add_patch(box)
    # Stage Header Pill
    pill_w = min(w - 3.0, max(24, len(label) * 0.92 + 3.0))
    pill_h = 2.4
    pill_x = x + 1.5
    pill_y = y + h - 1.2
    pill = patches.FancyBboxPatch(
        (pill_x, pill_y), pill_w, pill_h,
        boxstyle="round,pad=0.1,rounding_size=0.6",
        facecolor=label_bg, edgecolor=edgecolor, linewidth=1.0, zorder=zorder + 2
    )
    ax.add_patch(pill)
    ax.text(pill_x + pill_w / 2.0, pill_y + pill_h / 2.0, label,
            ha='center', va='center', fontsize=label_size, fontweight='bold',
            color='#FFFFFF', family='sans-serif', zorder=zorder + 3)


def draw_arrow(ax, x1, y1, x2, y2, color='#334155', lw=1.6, style='-|>', mutation=12, zorder=4):
    arr = patches.FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation,
        color=color, linewidth=lw, zorder=zorder,
        shrinkA=0, shrinkB=0
    )
    ax.add_patch(arr)


def draw_polyline_arrow(ax, pts, color='#475569', lw=1.3, mutation=11, zorder=4):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, linewidth=lw, solid_capstyle='round',
                solid_joinstyle='round', zorder=zorder)
    draw_arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], color=color, lw=lw, mutation=mutation, zorder=zorder)


def build_detailed_diagram():
    fig, ax = plt.subplots(figsize=(16, 22), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.8, 142)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')

    # Palette per stage
    C_ACQ = ('#EFF6FF', '#2563EB', '#DBEAFE')      # Blue
    C_PRE = ('#F0FDFA', '#0D9488', '#CCFBF1')      # Teal
    C_EDA = ('#F5F3FF', '#7C3AED', '#EDE9FE')      # Purple
    C_ARI = ('#FFFBEB', '#D97706', '#FEF3C7')      # Amber
    C_LST = ('#ECFDF5', '#059669', '#D1FAE5')      # Emerald
    C_EVA = ('#FFF1F2', '#E11D48', '#FFE4E6')      # Crimson
    C_CON = ('#F1F5F9', '#1E293B', '#E2E8F0')      # Navy/Slate

    # Main Title
    ax.text(50, 139.5, "End-to-End AQI Forecasting System Architecture",
            ha='center', va='center', fontsize=15.5, fontweight='bold', color='#0F172A')
    ax.text(50, 137.2, "UCS321 Engineering Design (EST) — IndianOil CPCL Manali Refinery Corridor, Chennai (2016–2020)",
            ha='center', va='center', fontsize=10.0, color='#475569')

    # =========================================================================
    # STAGE 1: DATA ACQUISITION (y = 127.0 to 134.0)
    # =========================================================================
    draw_container(ax, 6, 127.0, 88, 7.2, "1. DATA ACQUISITION & AUDIT",
                   facecolor=C_ACQ[0], edgecolor=C_ACQ[1], label_bg=C_ACQ[1])
    draw_box(ax, 10, 128.2, 37, 4.2, "Raw CPCB Daily Datasets",
             "data/raw/city_day.csv & station_day.csv\n(Hourly data bypassed; 24-hr regulatory alignment)",
             facecolor='#FFFFFF', edgecolor=C_ACQ[1], title_size=8.8, sub_size=7.0)
    draw_box(ax, 53, 128.2, 37, 4.2, "Chennai City-Level Selection",
             "2016-01-01 to 2020-06-30 (N = 1,643 days)\n2015 excluded (82-day monitoring outage)",
             facecolor='#FFFFFF', edgecolor=C_ACQ[1], title_size=8.8, sub_size=7.0)
    draw_arrow(ax, 47.2, 130.3, 52.8, 130.3, color=C_ACQ[1], lw=1.8)

    # Connector Stage 1 -> Stage 2
    draw_arrow(ax, 50, 126.8, 50, 124.2, color='#334155', lw=2.0, mutation=14)

    # =========================================================================
    # STAGE 2: PREPROCESSING (9 Steps, y = 101.5 to 124.0)
    # =========================================================================
    draw_container(ax, 4, 101.5, 92, 22.5, "2. PREPROCESSING PIPELINE (STRICT CHRONOLOGICAL ORDER)",
                   facecolor=C_PRE[0], edgecolor=C_PRE[1], label_bg=C_PRE[1])

    pre_steps = [
        ("2.1 Load & Filter", "Extract City == 'Chennai'\nDrop 2015 outage period"),
        ("2.2 Calendar Reindex", "Continuous daily index\n2016-01-01 to 2020-06-30"),
        ("2.3 Validity Checks", "Physical lower check:\nValues < 0 set to NaN"),
        ("2.4 Sensor Anomaly Removal", "Flag TN004 CO fault\n(19–22 Sep 2016 -> NaN)"),
        ("2.5 Gap-Based Imputation", "<=3d: Linear interp.\n4–30d: DOY train median"),
        ("2.6 Outlier Handling", "Upper winsorization (99.5th)\nAQI spikes preserved intact"),
        ("2.7 Calendar Features", "Cyclical sin/cos DOY\n& Fourier harmonics (K=3)"),
        ("2.8 Chronological Split", "Train 16–18 | Val 19 | Test\n19–20 | Stress Mar–Jun 20"),
        ("2.9 Train-Only Scaling", "MinMaxScaler(0, 1)\nFit exclusively on Train"),
    ]
    xs_3 = [7.5, 37.0, 66.5]
    ys_3 = [117.5, 110.2, 102.9]
    bw3, bh3 = 26.0, 4.3

    for idx, (t, s) in enumerate(pre_steps):
        r, c = divmod(idx, 3)
        draw_box(ax, xs_3[c], ys_3[r], bw3, bh3, t, s,
                 facecolor='#FFFFFF', edgecolor=C_PRE[1], title_size=8.4, sub_size=6.8)

    # Arrows within rows and across rows (clean gutter routing)
    for r in range(3):
        draw_arrow(ax, xs_3[0] + bw3 + 0.2, ys_3[r] + bh3 / 2, xs_3[1] - 0.2, ys_3[r] + bh3 / 2, color=C_PRE[1], lw=1.4)
        draw_arrow(ax, xs_3[1] + bw3 + 0.2, ys_3[r] + bh3 / 2, xs_3[2] - 0.2, ys_3[r] + bh3 / 2, color=C_PRE[1], lw=1.4)
    # Row 1 (box 3) -> Row 2 (box 4)
    draw_polyline_arrow(ax, [
        (xs_3[2] + bw3 / 2, ys_3[0] - 0.15),
        (xs_3[2] + bw3 / 2, ys_3[0] - 1.45),
        (xs_3[0] + bw3 / 2, ys_3[0] - 1.45),
        (xs_3[0] + bw3 / 2, ys_3[1] + bh3 + 0.15)
    ], color=C_PRE[1], lw=1.3)
    # Row 2 (box 6) -> Row 3 (box 7)
    draw_polyline_arrow(ax, [
        (xs_3[2] + bw3 / 2, ys_3[1] - 0.15),
        (xs_3[2] + bw3 / 2, ys_3[1] - 1.45),
        (xs_3[0] + bw3 / 2, ys_3[1] - 1.45),
        (xs_3[0] + bw3 / 2, ys_3[2] + bh3 + 0.15)
    ], color=C_PRE[1], lw=1.3)

    # Connector Stage 2 -> Stage 3
    draw_arrow(ax, 50, 101.3, 50, 98.7, color='#334155', lw=2.0, mutation=14)

    # =========================================================================
    # STAGE 3: EDA & VISUALIZATION (8 Analyses, y = 82.0 to 98.5)
    # =========================================================================
    draw_container(ax, 4, 82.0, 92, 16.5, "3. EXPLORATORY DATA ANALYSIS & TIME-SERIES DIAGNOSTICS",
                   facecolor=C_EDA[0], edgecolor=C_EDA[1], label_bg=C_EDA[1])

    eda_steps = [
        ("3.1 Descriptive Stats", "Right-skewed AQI (skew 2.02)\nTrain mean 112.9 -> Test 95.7"),
        ("3.2 Trend Analysis", "Multi-year downward drift\n& 2020 lockdown drop"),
        ("3.3 Seasonality", "Strong annual cycle (Fs=0.57)\nNegligible weekly cycle"),
        ("3.4 Category Dist.", ">93% Satisfactory / Moderate\nRare Severe episodes (<1.5%)"),
        ("3.5 Correlations", "PM2.5 primary driver (r=0.88)\nNext-day lag-1 r = 0.606"),
        ("3.6 STL Decomposition", "Additive trend, annual seasonal\n& residual spike isolation"),
        ("3.7 ADF / KPSS Tests", "Non-stationary raw AQI\nd=1 achieves strict stationarity"),
        ("3.8 ACF / PACF", "PACF lag-1 cutoff (0.697)\nGuides ARIMA(p,1,q) orders"),
    ]
    xs_4 = [6.5, 29.0, 51.5, 74.0]
    ys_2 = [91.5, 83.8]
    bw4, bh4 = 19.8, 4.3

    for idx, (t, s) in enumerate(eda_steps):
        r, c = divmod(idx, 4)
        draw_box(ax, xs_4[c], ys_2[r], bw4, bh4, t, s,
                 facecolor='#FFFFFF', edgecolor=C_EDA[1], title_size=8.2, sub_size=6.6)

    for r in range(2):
        for c in range(3):
            draw_arrow(ax, xs_4[c] + bw4 + 0.2, ys_2[r] + bh4 / 2,
                       xs_4[c + 1] - 0.2, ys_2[r] + bh4 / 2, color=C_EDA[1], lw=1.3)
    draw_polyline_arrow(ax, [
        (xs_4[3] + bw4 / 2, ys_2[0] - 0.15),
        (xs_4[3] + bw4 / 2, ys_2[0] - 1.65),
        (xs_4[0] + bw4 / 2, ys_2[0] - 1.65),
        (xs_4[0] + bw4 / 2, ys_2[1] + bh4 + 0.15)
    ], color=C_EDA[1], lw=1.3)

    # Fork from Stage 3 -> Parallel Branches (y = 81.8 -> 77.5)
    ax.plot([50, 50], [81.8, 79.8], color='#334155', linewidth=2.0, zorder=4)
    ax.plot([26, 74], [79.8, 79.8], color='#334155', linewidth=2.0, zorder=4)
    draw_arrow(ax, 26, 79.8, 26, 77.7, color=C_ARI[1], lw=2.0, mutation=14)
    draw_arrow(ax, 74, 79.8, 74, 77.7, color=C_LST[1], lw=2.0, mutation=14)

    # =========================================================================
    # STAGE 4: TWO PARALLEL MODELING BRANCHES (y = 26.5 to 77.5)
    # =========================================================================
    # 4A. ARIMA BRANCH (Left: x=4..48)
    draw_container(ax, 4, 26.5, 44, 51.0, "4A. CLASSICAL BRANCH (ARIMA / ARIMAX)",
                   facecolor=C_ARI[0], edgecolor=C_ARI[1], label_bg=C_ARI[1])
    arima_steps = [
        ("Baselines", "Naive Persistence (Test RMSE 32.10)\n7-Day MA (42.44) & Seasonal Naive 365d (71.61)"),
        ("Grid Search (136 Configs)", "Systematic evaluation across Pure ARIMA,\nARIMA + Fourier Harmonics & Exogenous ARIMAX"),
        ("Fourier / ARIMAX Champion", "ARIMAX(2,1,1) + Fourier(K=3) + PM2.5\nTest RMSE: 25.97 | MAE: 18.45 | R²: 0.6657"),
        ("Residual Diagnostics", "Ljung–Box residual autocorrelation audit,\nnormality & heteroskedasticity checks"),
        ("Walk-Forward (175 Origins)", "Recursive multi-step ARIMA-Fourier (h=1 to 7)\nh=1: 32.25 | h=4: 47.83 | h=7: 50.35 RMSE"),
    ]
    bw_a, bh_a = 38.0, 6.2
    x_a = 7.0
    ys_a = [67.0, 57.5, 48.0, 38.5, 29.0]
    for i, (t, s) in enumerate(arima_steps):
        draw_box(ax, x_a, ys_a[i], bw_a, bh_a, t, s,
                 facecolor='#FFFFFF', edgecolor=C_ARI[1], title_size=8.8, sub_size=7.0)
        if i < len(arima_steps) - 1:
            draw_arrow(ax, x_a + bw_a / 2, ys_a[i] - 0.15, x_a + bw_a / 2, ys_a[i + 1] + bh_a + 0.15,
                       color=C_ARI[1], lw=1.6)

    # 4B. LSTM BRANCH (Right: x=52..96)
    draw_container(ax, 52, 26.5, 44, 51.0, "4B. DEEP LEARNING BRANCH (LSTM / HYBRID)",
                   facecolor=C_LST[0], edgecolor=C_LST[1], label_bg=C_LST[1])
    lstm_steps = [
        ("Sequence Windows", "Chronological sliding lookbacks L in {7, 14, 21, 30}"),
        ("Stage 1: Features x Lookback", "80 runs: CORE, ALL9, Univariate, Lags (5 seeds)"),
        ("Stage 2: Architecture Tournament", "40 runs: Stacked, BiLSTM, GRU, CNN-LSTM (Winner)"),
        ("Stage 3: Optuna Bayesian HPO", "60 trials tuning units, dropout, LR, Huber/MSE"),
        ("Stage 4: Final Champion Retrain", "5-seed CNN-LSTM ensemble on Train+Val (RMSE 27.63)"),
        ("MC Dropout Uncertainty", "100 stochastic passes (epistemic coverage 37.9%)"),
        ("Stage 5: Direct 7-Day Model", "Dense(7) head; Val Epoch 2; beats Persist at h=5..7"),
        ("Stage 6: Hybrid ARIMAX+LSTM", "ARIMAX baseline + Residual LSTM (Test RMSE 26.81)"),
    ]
    bw_l, bh_l = 38.0, 4.1
    x_l = 55.0
    ys_l = [70.2, 64.5, 58.8, 53.1, 47.4, 41.7, 36.0, 30.3]
    #Wait: let's adjust slightly so bottom box ends around 28.5
    ys_l = [70.2, 64.4, 58.6, 52.8, 47.0, 41.2, 35.4, 29.6]
    for i, (t, s) in enumerate(lstm_steps):
        draw_box(ax, x_l, ys_l[i], bw_l, bh_l, t, s,
                 facecolor='#FFFFFF', edgecolor=C_LST[1], title_size=8.3, sub_size=6.7)
        if i < len(lstm_steps) - 1:
            draw_arrow(ax, x_l + bw_l / 2, ys_l[i] - 0.15, x_l + bw_l / 2, ys_l[i + 1] + bh_l + 0.15,
                       color=C_LST[1], lw=1.4)

    # Merge Branches -> Stage 5 Evaluation (y = 26.3 -> 22.0)
    ax.plot([26, 26], [26.3, 24.3], color=C_ARI[1], linewidth=2.0, zorder=4)
    ax.plot([74, 74], [26.3, 24.3], color=C_LST[1], linewidth=2.0, zorder=4)
    ax.plot([26, 74], [24.3, 24.3], color='#334155', linewidth=2.0, zorder=4)
    draw_arrow(ax, 50, 24.3, 50, 22.2, color='#334155', lw=2.0, mutation=14)

    # =========================================================================
    # STAGE 5: EVALUATION (5 Boxes, y = 11.0 to 22.0)
    # =========================================================================
    draw_container(ax, 4, 11.0, 92, 11.0, "5. COMPREHENSIVE EVALUATION & BENCHMARKING",
                   facecolor=C_EVA[0], edgecolor=C_EVA[1], label_bg=C_EVA[1])

    eval_steps = [
        ("5.1 Metrics", "2x2 Grid: RMSE, MAE\nR², CPCB Acc & QWK"),
        ("5.2 DM Test", "HLN correction (h=1)\n+ Bonferroni alpha=0.005"),
        ("5.3 Alerts", "AQI >100 & >200\nPrecision, Recall & F1"),
        ("5.4 Intervals", "Split-conformal (96.7%)\nvs MC-Dropout (37.9%)"),
        ("5.5 Stress Test", "Lockdown: Pure ARIMA\n13.51 vs Persist 14.45"),
    ]
    xs_5 = [6.0, 24.2, 42.4, 60.6, 78.8]
    bw5, bh5 = 15.6, 6.2
    for c, (t, s) in enumerate(eval_steps):
        draw_box(ax, xs_5[c], 12.6, bw5, bh5, t, s,
                 facecolor='#FFFFFF', edgecolor=C_EVA[1], title_size=8.2, sub_size=6.6)
        if c < 4:
            draw_arrow(ax, xs_5[c] + bw5 + 0.2, 15.7, xs_5[c + 1] - 0.2, 15.7,
                       color=C_EVA[1], lw=1.3)

    # Connector Stage 5 -> Stage 6
    draw_arrow(ax, 50, 10.8, 50, 8.6, color='#334155', lw=2.0, mutation=14)

    # =========================================================================
    # STAGE 6: CONCLUSION (y = -0.6 to 8.4)
    # =========================================================================
    draw_container(ax, 6, -0.6, 88, 9.0, "6. CONCLUSION & POTENTIAL OPERATIONAL USE",
                   facecolor=C_CON[0], edgecolor=C_CON[1], label_bg=C_CON[1])
    draw_box(ax, 8.5, 0.5, 83, 6.0,
             "Champion 1-Day Model: ARIMAX(2,1,1) + Fourier(K=3) + PM2.5 (Test RMSE 25.97, MAE 18.45)",
             "• Exogenous PM2.5 cuts error across both paradigms (-15.1% Classical, -12.9% DL); Direct 7-Day LSTM beats persistence at h=5..7.\n"
             "• Structural Break: Pure ARIMA (13.51) & Persistence (14.45) lead during COVID lockdown; LSTMs over-predict (+7.84 / +10.55 bias).\n"
             "• Operational Context: Provides informational city-level early warnings with conformal bounds; fence-line & weather inputs needed next.",
             facecolor='#FFFFFF', edgecolor=C_CON[1], title_size=9.0, sub_size=7.2)

    out_path = os.path.join(FIGURES_DIR, 'flow_diagram.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    print(f"Saved detailed flow diagram: {out_path}")


def build_compact_diagram():
    fig, ax = plt.subplots(figsize=(9, 13.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')

    # Title
    ax.text(50, 97.2, "AQI Forecasting Pipeline Overview",
            ha='center', va='center', fontsize=14.5, fontweight='bold', color='#0F172A')
    ax.text(50, 94.6, "Chennai (CPCL Manali Zone) | Daily Series 2016–2020",
            ha='center', va='center', fontsize=9.5, color='#475569')

    # 1. Data Acquisition
    draw_box(ax, 14, 83.0, 72, 8.8, "1. Data Acquisition & Audit",
             "• CPCB daily city_day.csv & station_day.csv (Chennai city-level)\n"
             "• 2016-01-01 to 2020-06-30 (N = 1,643 days; 2015 outage excluded)",
             facecolor='#EFF6FF', edgecolor='#2563EB', lw=1.8, title_size=10.0, sub_size=7.8)
    draw_arrow(ax, 50, 82.8, 50, 79.4, color='#334155', lw=1.8, mutation=13)

    # 2. Preprocessing
    draw_box(ax, 14, 68.5, 72, 10.6, "2. Preprocessing Pipeline (Leakage-Free)",
             "• TN004 CO anomaly flagged | Gap-adaptive imputation (linear <=3d, DOY median 4–30d)\n"
             "• Upper-only 99.5th pct winsorization (AQI spikes preserved intact)\n"
             "• Cyclical DOY + Fourier K=3 | Chronological split & Train-only MinMaxScaler",
             facecolor='#F0FDFA', edgecolor='#0D9488', lw=1.8, title_size=10.0, sub_size=7.8)
    draw_arrow(ax, 50, 68.3, 50, 64.9, color='#334155', lw=1.8, mutation=13)

    # 3. EDA & Diagnostics
    draw_box(ax, 14, 54.5, 72, 10.1, "3. EDA & Time-Series Diagnostics",
             "• PM2.5 primary driver (r = 0.88, lag-1 r = 0.606); strong annual seasonality\n"
             "• STL decomposition & CPCB category profiling (>93% Satisfactory/Moderate)\n"
             "• ADF/KPSS stationarity (d=1) & ACF/PACF lag identification",
             facecolor='#F5F3FF', edgecolor='#7C3AED', lw=1.8, title_size=10.0, sub_size=7.8)

    # Fork to Parallel Modeling
    ax.plot([50, 50], [54.3, 51.8], color='#334155', lw=1.8, zorder=4)
    ax.plot([28, 72], [51.8, 51.8], color='#334155', lw=1.8, zorder=4)
    draw_arrow(ax, 28, 51.8, 28, 49.4, color='#D97706', lw=1.8, mutation=13)
    draw_arrow(ax, 72, 51.8, 72, 49.4, color='#059669', lw=1.8, mutation=13)

    # 4A & 4B Parallel Boxes
    draw_box(ax, 8, 33.5, 40, 15.6, "4A. Classical Track (ARIMA)",
             "• Baselines (Persistence, 7d-MA, 365d)\n"
             "• Grid Search (136 configs)\n"
             "• Champion: ARIMAX(2,1,1)+F3+PM2.5\n"
             "  (Test RMSE: 25.97 | MAE: 18.45)\n"
             "• 7-Day Recursive ARIMA-Fourier",
             facecolor='#FFFBEB', edgecolor='#D97706', lw=1.8, title_size=9.5, sub_size=7.4)

    draw_box(ax, 52, 33.5, 40, 15.6, "4B. Deep Learning Track (LSTM)",
             "• Stage 1–2: Grid & CNN-LSTM Winner\n"
             "• Stage 3: Optuna Bayesian HPO (60 trials)\n"
             "• Champion CNN-LSTM (Test RMSE: 27.63)\n"
             "• Hybrid ARIMAX+LSTM (Test RMSE: 26.81)\n"
             "• Direct 7-Day LSTM (Val Epoch 2)",
             facecolor='#ECFDF5', edgecolor='#059669', lw=1.8, title_size=9.5, sub_size=7.4)

    # Merge Branches
    ax.plot([28, 28], [33.3, 30.8], color='#D97706', lw=1.8, zorder=4)
    ax.plot([72, 72], [33.3, 30.8], color='#059669', lw=1.8, zorder=4)
    ax.plot([28, 72], [30.8, 30.8], color='#334155', lw=1.8, zorder=4)
    draw_arrow(ax, 50, 30.8, 50, 28.4, color='#334155', lw=1.8, mutation=13)

    # 5. Evaluation
    draw_box(ax, 14, 17.2, 72, 10.9, "5. Evaluation & Statistical Validation",
             "• 2x2 Benchmark | Diebold–Mariano (HLN + Bonferroni alpha=0.005)\n"
             "• Operational Alerts (AQI >100 & >200) | Split-Conformal Intervals (96.7%)\n"
             "• COVID Lockdown Stress Window (Pure ARIMA 13.51 vs Persist 14.45)",
             facecolor='#FFF1F2', edgecolor='#E11D48', lw=1.8, title_size=10.0, sub_size=7.7)
    draw_arrow(ax, 50, 17.0, 50, 13.6, color='#334155', lw=1.8, mutation=13)

    # 6. Conclusion
    draw_box(ax, 14, 2.5, 72, 10.8, "6. Conclusion & Operational Advisory",
             "• ARIMAX(2,1,1)+F3+PM2.5 is optimal for 1-day ambient forecasting\n"
             "• Direct 7-Day LSTM beats persistence at h=5..7; ARIMA-Fourier leads h=2..6\n"
             "• Early-warning tool with conformal bounds (city-level ambient scope)",
             facecolor='#F1F5F9', edgecolor='#1E293B', lw=1.8, title_size=10.0, sub_size=7.7)

    out_path = os.path.join(FIGURES_DIR, 'flow_diagram_compact.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    print(f"Saved compact flow diagram: {out_path}")


if __name__ == '__main__':
    build_detailed_diagram()
    build_compact_diagram()

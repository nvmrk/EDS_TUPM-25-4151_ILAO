# EDS_TUPM-25-4151_ILAO
## Engineering Data Systems Pipeline — ENV-02: Aerosol Optical Depth vs. Relative Humidity

**Course:** Computer Programming 1 | **Academic Year:** 2026  
**Student:** ILAO, TUPM-25-4151  
**Pillar:** 1 — Environmental Engineering  
**Topic:** ENV-02 — Aerosol Optical Depth (AOD) vs. Humidity  
**Dataset:** Riyadh Air Quality Telemetry 2022–2024  
**Unique Filter:** `city == 'Riyadh'` (geographic station isolation)

---

## Project Structure

```
EDS_TUPM-25-4151_ILAO/
├── main.py                  # Full OOP pipeline (AeroDataPipeline class)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── data/
│   ├── riyadh_air_quality_2021_2023.csv   # Raw source dataset
│   ├── dataset_original.csv               # Filtered Riyadh slice (auto-generated)
│   └── dataset_cleaned.csv                # Cleaned & enriched data (auto-generated)
└── outputs/
    ├── static_01_pm25_histogram.png       # PM2.5 distribution + KDE
    ├── static_02_comparative_boxplot.png  # High-RH vs Low-RH group comparison
    ├── static_03_scatter_regression.png   # PM2.5 vs RH scatter + OLS line
    ├── anim_01_monthly_pm25_trend.gif     # Monthly PM2.5 time-series animation
    └── anim_02_hourly_pm25_profile.gif    # Diurnal PM2.5 bar profile animation
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/nvmrk/EDS_TUPM-25-4151_ILAO.git
cd EDS_TUPM-25-4151_ILAO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place the dataset
Ensure `riyadh_air_quality_2021_2023.csv` is inside the `data/` folder.

### 4. Run the pipeline
```bash
python main.py
```

All outputs (cleaned CSVs + 5 plots) are generated automatically.

---

## Pipeline Architecture

```
AeroDataPipeline
├── Module 1: ingest_data()      → Load CSV, apply city='Riyadh' filter
├── Module 2: clean_data()       → Null removal, dedup, type casting, features
├── Module 3: analyze_data()     → NumPy statistics, Pearson r, comparative
├── Module 4: visualize_data()   → 3 static + 2 animated plots
└── Module 5: run_pipeline()     → Orchestrator + engineering report
```

---

## Key Statistical Results

| Variable | Mean | Std Dev | Skewness |
|---|---|---|---|
| PM2.5 (µg/m³) | ~76.94 | ~41.78 | ~0.003 |
| Humidity (%) | ~45.09 | ~20.23 | ~−0.006 |
| PM10 (µg/m³) | ~104.77 | ~54.75 | — |
| Temperature (°C) | ~29.96 | ~8.66 | — |

**Pearson r (PM2.5 vs Humidity):** ~−0.005 (negligible — mineral dust dominance)

---

## Engineering Interpretation

Riyadh's near-zero PM2.5–Humidity correlation is physically meaningful: the city's aerosol burden is dominated by **mineral desert dust** (geogenic PM) rather than hygroscopic secondary aerosols. Unlike coastal industrial cities where high humidity promotes hygroscopic particle growth and optical depth amplification, Riyadh's dust particles are hydrophobic, making RH a poor predictor of aerosol loading. This finding has direct implications for **satellite AOD retrieval algorithms** applied over arid regions, which often assume a humidity-dependent aerosol model.

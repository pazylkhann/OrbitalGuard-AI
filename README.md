# OrbitalGuard AI

**AI-Powered Space Debris Collision Risk Prediction System**

> Democratizing space safety for small satellite operators through accessible, affordable, AI-driven collision risk assessment.

---

## The Problem

There are **36,000+ tracked pieces of space debris** orbiting Earth at speeds up to 28,000 km/h. A collision with even a small fragment can destroy a satellite worth $50M–$500M. While large operators like SpaceX and ESA have dedicated teams and expensive tools (LeoLabs: $100K+/year, AGI: $50K+/year), **small operators — universities, startups, and space agencies of developing countries — have no affordable solution**.

They receive raw conjunction data from Space-Track.org but lack the expertise and tools to interpret it. **OrbitalGuard AI solves this.**

---

## Solution

OrbitalGuard AI is an **AI-powered collision risk prediction system** that:

1. **Ingests TLE (Two-Line Element) data** from public sources (CelesTrak, Space-Track.org)
2. **Propagates satellite orbits** using the SGP4 algorithm to predict future positions
3. **Detects conjunction events** (close approaches) between all tracked objects
4. **Predicts collision probability** using a trained Random Forest ML model
5. **Generates actionable alerts** with specific maneuver recommendations

### Key Differentiators

| Feature | OrbitalGuard AI | LeoLabs | AGI (Ansys) | Space-Track |
|---------|----------------|---------|-------------|-------------|
| Price | $99–$2,999/mo | $100K+/yr | $50K+/yr | Free (raw data) |
| AI/ML Core | Yes | Partial | No | No |
| Small Operators | Yes | No | No | Raw data only |
| Maneuver Recommendations | Yes (AI-based) | Yes | Yes | No |
| API Access | Yes | Yes | Limited | Limited |

---

## Architecture

```
TLE Data (CelesTrak/Space-Track)
         |
         v
[orbital_propagator.py]  -- SGP4 orbit propagation
         |
         v
[conjunction_detector.py] -- Close approach detection
         |
         v
[ml_model.py]            -- Random Forest collision probability
         |
         v
[alert_system.py]        -- Risk alerts + maneuver recommendations
         |
         v
[main.py]                -- Pipeline orchestration + JSON export
```

### AI Model Details

- **Algorithm**: Random Forest Classifier (50 decision trees)
- **Features**: 15 orbital and conjunction parameters
- **Training Data**: Synthetic CDM (Conjunction Data Messages) based on NASA statistics
- **Performance**: ~80% accuracy, ~68% precision, ~55% recall
- **Output**: Collision probability (Pc) + risk level (CRITICAL/HIGH/MEDIUM/LOW)

---

## Quick Start

### Requirements

- Python 3.8+
- NumPy

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/OrbitalGuard-AI.git
cd OrbitalGuard-AI
pip install -r requirements.txt
```

### Run the Demo

```bash
# Full pipeline demo (recommended)
python src/main.py

# Demo predictions only (fastest)
python src/main.py --demo

# Train model only
python src/main.py --train
```

### Expected Output

```
+==============================================================+
|          [*] ORBITALGUARD AI  v1.0.0                        |
|     AI-Powered Space Debris Collision Risk Prediction        |
+==============================================================+

STEP 1/6: Loading Satellite TLE Data
  Loaded 8 space objects from TLE catalog

STEP 2/6: Initializing SGP4 Orbital Propagators
  OK  ISS (ZARYA)    | LEO (Low Earth Orbit) | Alt: 413-422 km
  OK  STARLINK-1007  | LEO (Low Earth Orbit) | Alt: 546-548 km
  ...

STEP 3/6: Training AI Collision Risk Model
  Accuracy: 80.5% | Precision: 68.5% | Recall: 55.0%

STEP 4/6: Scanning for Conjunction Events (72-hour window)
  Checked 28 pairs, found 1 conjunctions

STEP 5/6: Running AI Risk Assessment
  [   ] TERRA <-> COSMOS 2251 DEB
     Miss Distance: 61.894 km | Pc: 1.57e-07 | Risk: LOW

STEP 6/6: Generating Reports & Alerts
  Full report exported to: data/conjunction_report.json
```

---

## Project Structure

```
OrbitalGuard-AI/
├── src/
│   ├── main.py                  # Main entry point & pipeline orchestration
│   ├── orbital_propagator.py    # TLE parser + SGP4 orbit propagation
│   ├── conjunction_detector.py  # Close approach detection + feature extraction
│   ├── ml_model.py              # Random Forest ML model for collision prediction
│   └── alert_system.py          # Alert generation + report formatting
├── models/
│   └── collision_risk_model.json  # Trained model metadata (auto-generated)
├── data/
│   └── conjunction_report.json    # Analysis output (auto-generated)
├── docs/
│   ├── TECHNICAL_DOCS.md          # Technical documentation
│   └── PITCHDECK.md               # Investor pitch deck
├── tests/
│   └── test_pipeline.py           # Unit tests
├── requirements.txt
└── README.md
```

---

## Business Model

**Target Market**: Small satellite operators — universities, startups, developing country space agencies

**Pricing Tiers**:
- **Free**: 3 satellites, basic alerts, 24h delay
- **Starter** ($99/mo): 10 satellites, email alerts, API access
- **Professional** ($499/mo): 100 satellites, real-time alerts, maneuver recommendations
- **Enterprise** ($2,999/mo): Unlimited, priority support, custom models

**Market Size**:
- TAM: $2.5B (Space Situational Awareness market by 2030)
- SAM: $500M (small/medium operators segment)
- SOM: $50M (realistic 5-year target)

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Orbital Mechanics | SGP4 algorithm (implemented from scratch) |
| ML Framework | Custom Random Forest (NumPy only, no external ML deps) |
| Data Sources | CelesTrak TLE, NASA CDM (open data) |
| Output Format | JSON, CLI |

---

## Data Sources

- **TLE Data**: [CelesTrak](https://celestrak.org/) — free, public satellite orbital data
- **Conjunction Data**: [Space-Track.org](https://www.space-track.org/) — NASA/USSPACECOM conjunction data messages
- **Space Debris Catalog**: [ESA DISCOS](https://discosweb.esoc.esa.int/) — European Space Agency debris database

---

## Roadmap

### v1.0 (Current — MVP)
- [x] SGP4 orbital propagation
- [x] Conjunction detection (brute-force + refinement)
- [x] Random Forest collision probability prediction
- [x] Alert system with maneuver recommendations
- [x] JSON report export

### v2.0 (Q2 2026)
- [ ] Real-time Space-Track.org API integration
- [ ] REST API with authentication
- [ ] Web dashboard
- [ ] Email/SMS alert notifications

### v3.0 (Q4 2026)
- [ ] Deep learning model (LSTM for trajectory prediction)
- [ ] Covariance matrix integration for more accurate Pc
- [ ] Multi-constellation support (GPS, Galileo, GLONASS)
- [ ] Insurance risk scoring API

---

## Team

Built for AEROO Space AI Competition 2026.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## References

1. Hoots, F.R. & Roehrich, R.L. (1980). *Models for Propagation of NORAD Element Sets*. Spacetrack Report No. 3.
2. NASA Orbital Debris Program Office. *Conjunction Data Messages (CDMs)*.
3. ESA Space Debris Office. *Annual Space Environment Report 2024*.
4. Kelso, T.S. CelesTrak. https://celestrak.org/

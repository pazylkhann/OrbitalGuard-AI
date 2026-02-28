# -*- coding: utf-8 -*-
"""
OrbitalGuard AI - Main Entry Point
AI-Powered Space Debris Collision Risk Prediction System

Usage:
    python main.py              # Run full demo with sample data
    python main.py --train      # Train ML model only
    python main.py --demo       # Run demo predictions

Author: OrbitalGuard AI Team
Version: 1.0.0
"""

import sys
import os
import datetime
import json
import argparse
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_propagator import TLEParser, SGP4Propagator, load_sample_tle_data
from conjunction_detector import ConjunctionDetector, generate_synthetic_conjunction_data
from ml_model import CollisionRiskPredictor, FEATURE_NAMES
from alert_system import AlertSystem


BANNER = """
+==============================================================+
|                                                              |
|          [*] ORBITALGUARD AI  v1.0.0                        |
|                                                              |
|     AI-Powered Space Debris Collision Risk Prediction        |
|     Protecting satellites with Machine Learning              |
|                                                              |
+==============================================================+
"""

SEVERITY_ICONS = {
    'CRITICAL': '[!!!]',
    'HIGH':     '[!! ]',
    'MEDIUM':   '[!  ]',
    'LOW':      '[   ]'
}


def step_banner(step: int, total: int, title: str):
    """Print a step banner."""
    print("\n" + "-" * 65)
    print("  STEP {}/{}: {}".format(step, total, title))
    print("-" * 65)


def run_full_demo():
    """
    Run the complete OrbitalGuard AI pipeline demonstration.

    Pipeline:
    1. Load TLE data (satellite orbital parameters)
    2. Initialize orbital propagators (SGP4)
    3. Train ML model on synthetic CDM data
    4. Detect conjunction events (close approaches)
    5. Predict collision probabilities with AI
    6. Generate alerts and recommendations
    7. Export report
    """
    print(BANNER)
    print("  Analysis Time: {} UTC".format(
        datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')))
    print("  Mode: Full Pipeline Demo\n")

    # ----------------------------------------------------------
    # STEP 1: Load TLE Data
    # ----------------------------------------------------------
    step_banner(1, 6, "Loading Satellite TLE Data")

    tle_content = load_sample_tle_data()
    parser = TLEParser()
    satellites = parser.parse_tle_file(tle_content)

    print("Loaded {} space objects from TLE catalog".format(len(satellites)))
    print("\nObject Catalog:")
    for sat in satellites:
        print("  [{:5d}] {}".format(sat['sat_num'], sat['name']))

    # ----------------------------------------------------------
    # STEP 2: Initialize Orbital Propagators
    # ----------------------------------------------------------
    step_banner(2, 6, "Initializing SGP4 Orbital Propagators")

    propagators = []
    for sat_data in satellites:
        try:
            prop = SGP4Propagator(sat_data)
            info = prop.get_orbital_info()
            propagators.append(prop)
            print("  OK  {:<30s} | {:<30s} | Alt: {:.0f}-{:.0f} km".format(
                info['name'], info['altitude_type'],
                info['perigee_km'], info['apogee_km']))
        except Exception as e:
            print("  ERR Failed to initialize {}: {}".format(sat_data['name'], e))

    print("\n{} propagators initialized".format(len(propagators)))

    # ----------------------------------------------------------
    # STEP 3: Train ML Model
    # ----------------------------------------------------------
    step_banner(3, 6, "Training AI Collision Risk Model")

    print("Generating synthetic Conjunction Data Messages (CDMs)...")
    X_train, y_train = generate_synthetic_conjunction_data(n_samples=2000)

    predictor = CollisionRiskPredictor()
    metrics = predictor.train(X_train, y_train)

    # Save model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'collision_risk_model.json')
    predictor.save_model(model_path)
    print("\nModel trained and saved to {}".format(model_path))
    print("  Accuracy: {:.1f}% | Precision: {:.1f}% | Recall: {:.1f}%".format(
        metrics['accuracy'] * 100, metrics['precision'] * 100, metrics['recall'] * 100))

    # ----------------------------------------------------------
    # STEP 4: Detect Conjunctions
    # ----------------------------------------------------------
    step_banner(4, 6, "Scanning for Conjunction Events (72-hour window)")

    detector = ConjunctionDetector(
        propagators=propagators,
        search_window_hours=72.0,
        time_step_seconds=120.0  # 2-minute steps for demo speed
    )

    start_time = datetime.datetime.utcnow()
    conjunctions = detector.detect_conjunctions(
        start_time=start_time,
        threshold_km=100.0
    )

    if not conjunctions:
        print("\nNo conjunction events detected in 72-hour window")
        print("(This is expected with sample TLE data - real catalog has 36,000+ objects)")
        print("\nGenerating synthetic conjunction scenarios for demonstration...")
        conjunctions = _create_demo_conjunctions(propagators, start_time)

    print("\nFound {} conjunction event(s)".format(len(conjunctions)))

    # ----------------------------------------------------------
    # STEP 5: AI Risk Assessment
    # ----------------------------------------------------------
    step_banner(5, 6, "Running AI Risk Assessment")

    alert_system = AlertSystem()
    predictions = []
    alerts = []

    for conj in conjunctions:
        features = detector.extract_features(conj)
        prediction = predictor.predict(features)
        conj.collision_probability = prediction.collision_probability
        conj.risk_level = prediction.risk_level

        predictions.append(prediction)

        alert = alert_system.generate_alert(conj, prediction)
        alerts.append(alert)

        icon = SEVERITY_ICONS.get(prediction.risk_level, '[?]')
        print("\n  {} {} <-> {}".format(icon, conj.sat1_name, conj.sat2_name))
        print("     Miss Distance: {:.3f} km | Pc: {:.2e} | Risk: {}".format(
            conj.miss_distance_km, prediction.collision_probability, prediction.risk_level))

    # ----------------------------------------------------------
    # STEP 6: Generate Reports
    # ----------------------------------------------------------
    step_banner(6, 6, "Generating Reports & Alerts")

    high_risk = [a for a in alerts if a.severity in ('CRITICAL', 'HIGH')]
    if high_risk:
        print("\n" + "!" * 65)
        print("  {} HIGH-PRIORITY ALERT(S) REQUIRE ATTENTION:".format(len(high_risk)))
        print("!" * 65)
        for alert in high_risk:
            print("\n" + alert_system.format_alert(alert))

    print("\n" + alert_system.generate_summary_report(conjunctions, predictions))

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, 'conjunction_report.json')
    alert_system.export_to_json(alerts, output_file)

    print("\nFull report exported to: {}".format(output_file))
    print("\n" + "=" * 65)
    print("  OrbitalGuard AI Analysis Complete!")
    print("  Analyzed {} objects | Found {} conjunctions | {} require action".format(
        len(propagators), len(conjunctions), len(high_risk)))
    print("=" * 65 + "\n")

    return conjunctions, predictions, alerts


def _create_demo_conjunctions(propagators, start_time):
    """Create synthetic conjunction events for demonstration."""
    from conjunction_detector import ConjunctionEvent

    demo_conjunctions = []

    scenarios = [
        {
            'sat1_idx': 0, 'sat2_idx': 4,
            'miss_distance': 2.3,
            'rel_velocity': 11.8,
            'hours_ahead': 18.5,
            'description': 'ISS vs Cosmos 2251 Debris'
        },
        {
            'sat1_idx': 1, 'sat2_idx': 5,
            'miss_distance': 0.8,
            'rel_velocity': 14.2,
            'hours_ahead': 6.2,
            'description': 'Starlink vs Fengyun 1C Debris'
        },
        {
            'sat1_idx': 6, 'sat2_idx': 7,
            'miss_distance': 45.0,
            'rel_velocity': 0.3,
            'hours_ahead': 52.0,
            'description': 'Sentinel-2A vs Landsat 8'
        }
    ]

    for scenario in scenarios:
        if (scenario['sat1_idx'] >= len(propagators) or
                scenario['sat2_idx'] >= len(propagators)):
            continue

        prop1 = propagators[scenario['sat1_idx']]
        prop2 = propagators[scenario['sat2_idx']]

        tca = start_time + datetime.timedelta(hours=scenario['hours_ahead'])

        try:
            pos1, vel1 = prop1.propagate(tca)
            pos2, vel2 = prop2.propagate(tca)
        except Exception:
            pos1 = np.array([7000.0, 0.0, 0.0])
            pos2 = pos1 + np.array([scenario['miss_distance'], 0.0, 0.0])
            vel1 = np.array([0.0, 7.5, 0.0])
            vel2 = np.array([0.0, -7.5 + scenario['rel_velocity'], 0.0])

        direction = np.array([1.0, 0.0, 0.0])
        pos2 = pos1 + direction * scenario['miss_distance']

        alt1 = max(0, np.linalg.norm(pos1) - 6378.137)
        alt2 = max(0, np.linalg.norm(pos2) - 6378.137)

        conj = ConjunctionEvent(
            sat1_name=prop1.tle['name'],
            sat2_name=prop2.tle['name'],
            sat1_num=prop1.tle['sat_num'],
            sat2_num=prop2.tle['sat_num'],
            time_of_closest_approach=tca,
            miss_distance_km=scenario['miss_distance'],
            relative_velocity_km_s=scenario['rel_velocity'],
            sat1_position=pos1,
            sat2_position=pos2,
            sat1_velocity=vel1,
            sat2_velocity=vel2,
            sat1_altitude_km=alt1,
            sat2_altitude_km=alt2,
            approach_angle_deg=95.0 if scenario['rel_velocity'] > 5 else 15.0,
            time_to_conjunction_hours=scenario['hours_ahead']
        )

        demo_conjunctions.append(conj)
        print("  + Demo scenario: {}".format(scenario['description']))

    return demo_conjunctions


def run_training_only():
    """Train and evaluate the ML model."""
    print(BANNER)
    print("  Mode: Model Training\n")

    print("Generating synthetic CDM training data...")
    X, y = generate_synthetic_conjunction_data(n_samples=3000)

    predictor = CollisionRiskPredictor()
    metrics = predictor.train(X, y)

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'collision_risk_model.json')
    predictor.save_model(model_path)

    print("\nModel saved to {}".format(model_path))
    return metrics


def run_demo_predictions():
    """Run demo predictions on predefined scenarios."""
    print(BANNER)
    print("  Mode: Demo Predictions\n")

    predictor = CollisionRiskPredictor()

    X, y = generate_synthetic_conjunction_data(n_samples=1000)
    predictor.train(X, y)

    scenarios = [
        {
            'name': 'CRITICAL: Head-on collision course',
            'features': [0.3, 14.5, 550, 549, 1, 178, 2.5, 0, 7.6, 7.6, 0.2, 0.2, 0.1, 4.35, 3.33]
        },
        {
            'name': 'HIGH: Close approach at high velocity',
            'features': [2.5, 12.3, 550, 548, 2, 95, 18.5, 0, 7.6, 7.6, 1.8, 1.7, 0.3, 30.75, 0.4]
        },
        {
            'name': 'MEDIUM: Moderate risk conjunction',
            'features': [15.0, 7.2, 800, 795, 5, 45, 36.0, 0, 7.4, 7.4, 10.0, 10.0, 5.0, 108.0, 0.067]
        },
        {
            'name': 'LOW: Distant pass',
            'features': [75.0, 2.1, 1200, 1190, 10, 20, 60.0, 0, 7.2, 7.2, 50.0, 50.0, 25.0, 157.5, 0.013]
        }
    ]

    print("=" * 65)
    print("  DEMO COLLISION RISK PREDICTIONS")
    print("=" * 65 + "\n")

    for scenario in scenarios:
        features = np.array(scenario['features'], dtype=np.float32)
        prediction = predictor.predict(features)

        icon = SEVERITY_ICONS.get(prediction.risk_level, '[?]')
        print("{} Scenario: {}".format(icon, scenario['name']))
        print("   Collision Probability: {:.2e} ({:.4f}%)".format(
            prediction.collision_probability,
            prediction.collision_probability * 100))
        print("   Risk Level: {} | Confidence: {:.0f}%".format(
            prediction.risk_level, prediction.confidence * 100))
        print("   Recommendation: {}".format(prediction.recommendation[:100]))
        print()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='OrbitalGuard AI - Space Debris Collision Risk Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run full demo pipeline
  python main.py --train      # Train ML model only
  python main.py --demo       # Run demo predictions
        """
    )

    parser.add_argument('--train', action='store_true', help='Train ML model only')
    parser.add_argument('--demo', action='store_true', help='Run demo predictions')

    args = parser.parse_args()

    if args.train:
        run_training_only()
    elif args.demo:
        run_demo_predictions()
    else:
        run_full_demo()


if __name__ == "__main__":
    main()

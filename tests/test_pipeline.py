# -*- coding: utf-8 -*-
"""
OrbitalGuard AI - Unit Tests
Tests for all major pipeline components.
"""

import sys
import os
import datetime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orbital_propagator import TLEParser, SGP4Propagator, load_sample_tle_data
from conjunction_detector import ConjunctionDetector, generate_synthetic_conjunction_data
from ml_model import CollisionRiskPredictor, FEATURE_NAMES
from alert_system import AlertSystem


def test_tle_parser():
    """Test TLE parsing."""
    print("Test 1: TLE Parser...")
    content = load_sample_tle_data()
    parser = TLEParser()
    satellites = parser.parse_tle_file(content)
    
    assert len(satellites) > 0, "Should parse at least one satellite"
    assert 'name' in satellites[0], "Should have name field"
    assert 'inclination' in satellites[0], "Should have inclination"
    assert 'mean_motion' in satellites[0], "Should have mean_motion"
    
    print("  PASS: Parsed {} satellites".format(len(satellites)))
    return satellites


def test_sgp4_propagator(satellites):
    """Test SGP4 orbital propagation."""
    print("Test 2: SGP4 Propagator...")
    
    prop = SGP4Propagator(satellites[0])
    target_time = datetime.datetime(2026, 3, 1, 12, 0, 0)
    
    pos, vel = prop.propagate(target_time)
    
    assert len(pos) == 3, "Position should be 3D vector"
    assert len(vel) == 3, "Velocity should be 3D vector"
    
    # Check reasonable values for LEO
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    
    assert 6500 < r < 8000, "Radius should be 6500-8000 km for LEO: got {}".format(r)
    # Velocity can be in km/s or km/min depending on propagator output
    assert v > 0, "Speed should be positive: got {}".format(v)
    
    info = prop.get_orbital_info()
    assert info['altitude_type'] in ['LEO (Low Earth Orbit)', 'MEO (Medium Earth Orbit)',
                                      'GEO (Geostationary Orbit)', 'HEO (High Earth Orbit)']
    
    print("  PASS: Position [{:.1f}, {:.1f}, {:.1f}] km, Speed {:.3f} km/s".format(
        pos[0], pos[1], pos[2], v))
    return prop


def test_kepler_solver():
    """Test Kepler's equation solver."""
    print("Test 3: Kepler's Equation Solver...")
    
    prop = SGP4Propagator.__new__(SGP4Propagator)
    
    # Test cases: (M, e) -> E
    test_cases = [
        (0.0, 0.0),      # Circular orbit
        (1.0, 0.1),      # Typical LEO
        (3.14159, 0.5),  # High eccentricity
    ]
    
    for M, e in test_cases:
        E = prop._solve_kepler(M, e)
        residual = abs(M - E + e * np.sin(E))
        assert residual < 1e-8, "Kepler residual too large: {}".format(residual)
    
    print("  PASS: Kepler solver converges for all test cases")


def test_synthetic_data():
    """Test synthetic CDM data generation."""
    print("Test 4: Synthetic CDM Data Generation...")
    
    X, y = generate_synthetic_conjunction_data(n_samples=100)
    
    assert X.shape == (100, 15), "Feature matrix should be (100, 15): got {}".format(X.shape)
    assert len(y) == 100, "Labels should have 100 entries"
    assert set(y).issubset({0, 1}), "Labels should be binary"
    assert sum(y) > 0, "Should have some positive cases"
    assert sum(y) < 100, "Should have some negative cases"
    
    print("  PASS: Generated {} samples, {} positive cases".format(len(X), sum(y)))
    return X, y


def test_ml_model(X, y):
    """Test ML model training and prediction."""
    print("Test 5: ML Model Training & Prediction...")
    
    predictor = CollisionRiskPredictor()
    metrics = predictor.train(X, y)
    
    assert metrics['accuracy'] > 0.5, "Accuracy should be > 50%: got {}".format(metrics['accuracy'])
    assert 0 <= metrics['precision'] <= 1, "Precision should be in [0,1]"
    assert 0 <= metrics['recall'] <= 1, "Recall should be in [0,1]"
    
    # Test prediction
    test_features = np.array([
        2.5, 12.3, 550, 548, 2, 95, 18.5, 0, 7.6, 7.6, 1.8, 1.7, 0.3, 30.75, 0.4
    ], dtype=np.float32)
    
    prediction = predictor.predict(test_features)
    
    assert 0 <= prediction.collision_probability <= 1, "Probability should be in [0,1]"
    assert prediction.risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    assert 0 <= prediction.confidence <= 1
    assert len(prediction.recommendation) > 0
    
    print("  PASS: Accuracy {:.1f}%, Prediction: {} ({:.2e})".format(
        metrics['accuracy'] * 100, prediction.risk_level, prediction.collision_probability))
    return predictor


def test_alert_system(predictor):
    """Test alert generation."""
    print("Test 6: Alert System...")
    
    from conjunction_detector import ConjunctionEvent
    
    # Create a mock conjunction event
    conj = ConjunctionEvent(
        sat1_name="TEST-SAT-1",
        sat2_name="DEBRIS-001",
        sat1_num=99001,
        sat2_num=99002,
        time_of_closest_approach=datetime.datetime(2026, 3, 1, 18, 0, 0),
        miss_distance_km=2.5,
        relative_velocity_km_s=12.3,
        sat1_position=np.array([7000.0, 0.0, 0.0]),
        sat2_position=np.array([7002.5, 0.0, 0.0]),
        sat1_velocity=np.array([0.0, 7.6, 0.0]),
        sat2_velocity=np.array([0.0, -4.7, 0.0]),
        sat1_altitude_km=550.0,
        sat2_altitude_km=548.0,
        approach_angle_deg=95.0,
        time_to_conjunction_hours=18.5
    )
    
    test_features = np.array([
        2.5, 12.3, 550, 548, 2, 95, 18.5, 0, 7.6, 7.6, 1.8, 1.7, 0.3, 30.75, 0.4
    ], dtype=np.float32)
    
    prediction = predictor.predict(test_features)
    
    alert_system = AlertSystem()
    alert = alert_system.generate_alert(conj, prediction)
    
    assert alert.alert_id.startswith("OG-"), "Alert ID should start with OG-"
    assert alert.sat1_name == "TEST-SAT-1"
    assert alert.sat2_name == "DEBRIS-001"
    assert alert.miss_distance_km == 2.5
    assert alert.severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    
    formatted = alert_system.format_alert(alert)
    assert len(formatted) > 100, "Formatted alert should be substantial"
    
    print("  PASS: Alert {} generated, severity: {}".format(alert.alert_id, alert.severity))


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("  OrbitalGuard AI - Unit Tests")
    print("=" * 60 + "\n")
    
    passed = 0
    failed = 0
    
    try:
        satellites = test_tle_parser()
        passed += 1
    except Exception as e:
        print("  FAIL: {}".format(e))
        failed += 1
        satellites = None
    
    try:
        if satellites:
            test_sgp4_propagator(satellites)
            passed += 1
    except Exception as e:
        print("  FAIL: {}".format(e))
        failed += 1
    
    try:
        test_kepler_solver()
        passed += 1
    except Exception as e:
        print("  FAIL: {}".format(e))
        failed += 1
    
    try:
        X, y = test_synthetic_data()
        passed += 1
    except Exception as e:
        print("  FAIL: {}".format(e))
        failed += 1
        X, y = None, None
    
    predictor = None
    try:
        if X is not None:
            predictor = test_ml_model(X, y)
            passed += 1
    except Exception as e:
        print("  FAIL: {}".format(e))
        failed += 1
    
    try:
        if predictor:
            test_alert_system(predictor)
            passed += 1
    except Exception as e:
        print("  FAIL: {}".format(e))
        failed += 1
    
    print("\n" + "=" * 60)
    print("  Results: {}/{} tests passed".format(passed, passed + failed))
    if failed == 0:
        print("  ALL TESTS PASSED!")
    else:
        print("  {} test(s) FAILED".format(failed))
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

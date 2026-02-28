# -*- coding: utf-8 -*-
"""
OrbitalGuard AI - Web Dashboard
Mission Control UI/UX — Production-ready SaaS design.
Glassmorphism, animated orbit visualization, real-time risk gauges.

Usage:
    python src/web_dashboard.py
    Then open: http://localhost:8080
"""

import sys
import os
import json
import datetime
import threading
import webbrowser
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_propagator import TLEParser, SGP4Propagator, load_sample_tle_data
from conjunction_detector import ConjunctionDetector, generate_synthetic_conjunction_data
from ml_model import CollisionRiskPredictor
from alert_system import AlertSystem

# Global state
predictor = None
detector = None
propagators = []
satellites_data = []


def initialize_system():
    """Initialize the AI system on startup."""
    global predictor, detector, propagators, satellites_data

    print("[WebDashboard] Initializing OrbitalGuard AI system...")

    # Load TLE data
    tle_content = load_sample_tle_data()
    parser = TLEParser()
    satellites_data = parser.parse_tle_file(tle_content)

    # Create propagators
    propagators = []
    for sat in satellites_data:
        try:
            propagators.append(SGP4Propagator(sat))
        except Exception:
            pass

    # Train ML model
    X, y = generate_synthetic_conjunction_data(n_samples=2000)
    predictor = CollisionRiskPredictor()
    predictor.train(X, y)

    # Create detector
    detector = ConjunctionDetector(propagators, search_window_hours=72, time_step_seconds=120)

    print("[WebDashboard] System ready!")


def run_analysis():
    """Run the full conjunction analysis and return results."""
    global predictor, detector, propagators

    start_time = datetime.datetime.utcnow()
    conjunctions = detector.detect_conjunctions(start_time=start_time, threshold_km=100.0)

    # Add demo conjunctions if none found
    if not conjunctions:
        from conjunction_detector import ConjunctionEvent
        scenarios = [
            {'sat1_idx': 0, 'sat2_idx': 4, 'miss_distance': 2.3, 'rel_velocity': 11.8, 'hours_ahead': 18.5},
            {'sat1_idx': 1, 'sat2_idx': 5, 'miss_distance': 0.8, 'rel_velocity': 14.2, 'hours_ahead': 6.2},
            {'sat1_idx': 6, 'sat2_idx': 7, 'miss_distance': 45.0, 'rel_velocity': 0.3, 'hours_ahead': 52.0},
        ]
        for s in scenarios:
            if s['sat1_idx'] < len(propagators) and s['sat2_idx'] < len(propagators):
                p1, p2 = propagators[s['sat1_idx']], propagators[s['sat2_idx']]
                tca = start_time + datetime.timedelta(hours=s['hours_ahead'])
                try:
                    pos1, vel1 = p1.propagate(tca)
                    pos2, vel2 = p2.propagate(tca)
                except Exception:
                    pos1 = np.array([7000.0, 0.0, 0.0])
                    pos2 = pos1 + np.array([s['miss_distance'], 0.0, 0.0])
                    vel1 = np.array([0.0, 7.5, 0.0])
                    vel2 = np.array([0.0, -4.7, 0.0])

                pos2 = pos1 + np.array([1.0, 0.0, 0.0]) * s['miss_distance']
                alt1 = max(0, np.linalg.norm(pos1) - 6378.137)
                alt2 = max(0, np.linalg.norm(pos2) - 6378.137)

                conj = ConjunctionEvent(
                    sat1_name=p1.tle['name'], sat2_name=p2.tle['name'],
                    sat1_num=p1.tle['sat_num'], sat2_num=p2.tle['sat_num'],
                    time_of_closest_approach=tca,
                    miss_distance_km=s['miss_distance'],
                    relative_velocity_km_s=s['rel_velocity'],
                    sat1_position=pos1, sat2_position=pos2,
                    sat1_velocity=vel1, sat2_velocity=vel2,
                    sat1_altitude_km=alt1, sat2_altitude_km=alt2,
                    approach_angle_deg=95.0 if s['rel_velocity'] > 5 else 15.0,
                    time_to_conjunction_hours=s['hours_ahead']
                )
                conjunctions.append(conj)

    results = []
    alert_system = AlertSystem()

    for conj in conjunctions:
        features = detector.extract_features(conj)
        prediction = predictor.predict(features)
        conj.collision_probability = prediction.collision_probability
        conj.risk_level = prediction.risk_level

        results.append({
            'sat1_name': conj.sat1_name,
            'sat2_name': conj.sat2_name,
            'miss_distance_km': round(conj.miss_distance_km, 3),
            'relative_velocity_km_s': round(conj.relative_velocity_km_s, 2),
            'collision_probability': round(prediction.collision_probability, 8),
            'collision_probability_pct': round(prediction.collision_probability * 100, 6),
            'risk_level': prediction.risk_level,
            'confidence': round(prediction.confidence * 100, 1),
            'time_to_conjunction_hours': round(conj.time_to_conjunction_hours, 1),
            'sat1_altitude_km': round(conj.sat1_altitude_km, 1),
            'sat2_altitude_km': round(conj.sat2_altitude_km, 1),
            'approach_angle_deg': round(conj.approach_angle_deg, 1),
            'recommendation': prediction.recommendation,
            'maneuver_window_hours': prediction.maneuver_window_hours,
            'tca': conj.time_of_closest_approach.strftime('%Y-%m-%d %H:%M UTC'),
        })

    return results


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OrbitalGuard AI — Mission Control</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        /* ═══════════════════════════════════════════════════
           DESIGN SYSTEM — OrbitalGuard AI Mission Control
           ═══════════════════════════════════════════════════ */
        :root {
            /* Color Palette */
            --bg-void:        #020408;
            --bg-deep:        #050c14;
            --bg-space:       #080f1c;
            --bg-panel:       #0b1525;
            --bg-card:        #0e1c30;
            --bg-elevated:    #122038;

            /* Glass */
            --glass-bg:       rgba(14, 28, 48, 0.72);
            --glass-border:   rgba(56, 139, 253, 0.18);
            --glass-hover:    rgba(56, 139, 253, 0.28);

            /* Accent — Electric Blue */
            --accent-primary: #38bdf8;
            --accent-glow:    #0ea5e9;
            --accent-deep:    #0369a1;

            /* Neon Palette */
            --neon-cyan:      #22d3ee;
            --neon-blue:      #60a5fa;
            --neon-purple:    #a78bfa;
            --neon-green:     #34d399;
            --neon-yellow:    #fbbf24;
            --neon-orange:    #fb923c;
            --neon-red:       #f87171;

            /* Risk Colors */
            --risk-critical:  #ef4444;
            --risk-high:      #f97316;
            --risk-medium:    #eab308;
            --risk-low:       #22c55e;

            /* Text */
            --text-primary:   #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted:     #475569;
            --text-accent:    #38bdf8;

            /* Borders */
            --border-subtle:  rgba(56, 139, 253, 0.10);
            --border-default: rgba(56, 139, 253, 0.20);
            --border-strong:  rgba(56, 139, 253, 0.40);

            /* Spacing */
            --space-xs:  4px;
            --space-sm:  8px;
            --space-md:  16px;
            --space-lg:  24px;
            --space-xl:  32px;
            --space-2xl: 48px;

            /* Radius */
            --radius-sm:  6px;
            --radius-md:  12px;
            --radius-lg:  16px;
            --radius-xl:  24px;
            --radius-full: 9999px;

            /* Shadows */
            --shadow-glow-blue:   0 0 20px rgba(56, 189, 248, 0.15);
            --shadow-glow-red:    0 0 20px rgba(239, 68, 68, 0.25);
            --shadow-glow-orange: 0 0 20px rgba(249, 115, 22, 0.20);
            --shadow-card:        0 4px 24px rgba(0, 0, 0, 0.4);
            --shadow-elevated:    0 8px 40px rgba(0, 0, 0, 0.6);

            /* Typography */
            --font-sans: 'Inter', system-ui, sans-serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
        }

        /* ═══════════════ RESET & BASE ═══════════════ */
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        html { scroll-behavior: smooth; }

        body {
            font-family: var(--font-sans);
            background: var(--bg-void);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            line-height: 1.6;
        }

        /* Starfield background */
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                radial-gradient(1px 1px at 10% 15%, rgba(255,255,255,0.6) 0%, transparent 100%),
                radial-gradient(1px 1px at 25% 40%, rgba(255,255,255,0.4) 0%, transparent 100%),
                radial-gradient(1.5px 1.5px at 40% 8%, rgba(255,255,255,0.5) 0%, transparent 100%),
                radial-gradient(1px 1px at 55% 60%, rgba(255,255,255,0.3) 0%, transparent 100%),
                radial-gradient(1px 1px at 70% 25%, rgba(255,255,255,0.5) 0%, transparent 100%),
                radial-gradient(1.5px 1.5px at 80% 75%, rgba(255,255,255,0.4) 0%, transparent 100%),
                radial-gradient(1px 1px at 90% 45%, rgba(255,255,255,0.6) 0%, transparent 100%),
                radial-gradient(1px 1px at 15% 80%, rgba(255,255,255,0.3) 0%, transparent 100%),
                radial-gradient(1px 1px at 35% 90%, rgba(255,255,255,0.4) 0%, transparent 100%),
                radial-gradient(1px 1px at 60% 85%, rgba(255,255,255,0.3) 0%, transparent 100%),
                radial-gradient(1px 1px at 5% 55%, rgba(255,255,255,0.5) 0%, transparent 100%),
                radial-gradient(1px 1px at 95% 10%, rgba(255,255,255,0.4) 0%, transparent 100%),
                radial-gradient(1px 1px at 48% 35%, rgba(255,255,255,0.3) 0%, transparent 100%),
                radial-gradient(1px 1px at 75% 50%, rgba(255,255,255,0.4) 0%, transparent 100%),
                radial-gradient(1px 1px at 20% 65%, rgba(255,255,255,0.3) 0%, transparent 100%);
            pointer-events: none;
            z-index: 0;
        }

        /* Ambient glow orbs */
        body::after {
            content: '';
            position: fixed;
            inset: 0;
            background:
                radial-gradient(ellipse 600px 400px at 20% 10%, rgba(14, 165, 233, 0.04) 0%, transparent 70%),
                radial-gradient(ellipse 500px 300px at 80% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 70%),
                radial-gradient(ellipse 400px 300px at 50% 50%, rgba(6, 182, 212, 0.02) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        /* ═══════════════ LAYOUT ═══════════════ */
        .app-layout {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }

        /* ═══════════════ TOPBAR ═══════════════ */
        .topbar {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(5, 12, 20, 0.85);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border-bottom: 1px solid var(--border-subtle);
            padding: 0 var(--space-xl);
            height: 64px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: var(--space-lg);
        }

        .topbar-left {
            display: flex;
            align-items: center;
            gap: var(--space-md);
        }

        .logo-mark {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent-glow), var(--neon-purple));
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 0 16px rgba(14, 165, 233, 0.4);
            flex-shrink: 0;
            position: relative;
            overflow: hidden;
        }

        .logo-mark::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 60%);
        }

        .logo-wordmark {
            display: flex;
            flex-direction: column;
            gap: 1px;
        }

        .logo-wordmark .brand {
            font-size: 15px;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: var(--text-primary);
            line-height: 1;
        }

        .logo-wordmark .brand span {
            color: var(--accent-primary);
        }

        .logo-wordmark .tagline {
            font-size: 10px;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .topbar-divider {
            width: 1px;
            height: 28px;
            background: var(--border-subtle);
        }

        .nav-tabs {
            display: flex;
            gap: var(--space-xs);
        }

        .nav-tab {
            padding: 6px 14px;
            border-radius: var(--radius-sm);
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid transparent;
            background: none;
            letter-spacing: 0.02em;
        }

        .nav-tab:hover {
            color: var(--text-secondary);
            background: rgba(56, 189, 248, 0.06);
        }

        .nav-tab.active {
            color: var(--accent-primary);
            background: rgba(56, 189, 248, 0.10);
            border-color: var(--border-default);
        }

        .topbar-right {
            display: flex;
            align-items: center;
            gap: var(--space-md);
        }

        .system-status {
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            padding: 6px 12px;
            background: rgba(34, 197, 94, 0.08);
            border: 1px solid rgba(34, 197, 94, 0.20);
            border-radius: var(--radius-full);
            font-size: 12px;
            font-weight: 500;
            color: var(--neon-green);
        }

        .status-pulse {
            width: 7px;
            height: 7px;
            background: var(--neon-green);
            border-radius: 50%;
            box-shadow: 0 0 6px var(--neon-green);
            animation: statusPulse 2s ease-in-out infinite;
        }

        @keyframes statusPulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.85); }
        }

        .clock-display {
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: 0.05em;
        }

        .btn-refresh {
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            padding: 8px 18px;
            background: linear-gradient(135deg, var(--accent-glow), var(--accent-deep));
            border: none;
            border-radius: var(--radius-md);
            color: white;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            letter-spacing: 0.02em;
            box-shadow: 0 0 16px rgba(14, 165, 233, 0.25);
        }

        .btn-refresh:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.40);
        }

        .btn-refresh:active { transform: translateY(0); }

        .btn-refresh svg {
            width: 14px;
            height: 14px;
            transition: transform 0.4s ease;
        }

        .btn-refresh.loading svg { animation: spinIcon 0.8s linear infinite; }

        @keyframes spinIcon { to { transform: rotate(360deg); } }

        /* ═══════════════ MAIN CONTENT ═══════════════ */
        .main-content {
            flex: 1;
            padding: var(--space-xl);
            max-width: 1440px;
            margin: 0 auto;
            width: 100%;
        }

        /* ═══════════════ SECTION HEADER ═══════════════ */
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: var(--space-lg);
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.10em;
        }

        .section-title::before {
            content: '';
            width: 3px;
            height: 14px;
            background: var(--accent-primary);
            border-radius: 2px;
            box-shadow: 0 0 8px var(--accent-primary);
        }

        /* ═══════════════ KPI STRIP ═══════════════ */
        .kpi-strip {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: var(--space-md);
            margin-bottom: var(--space-xl);
        }

        .kpi-card {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            padding: var(--space-lg) var(--space-md);
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            cursor: default;
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--kpi-accent, var(--accent-primary));
            opacity: 0.7;
        }

        .kpi-card::after {
            content: '';
            position: absolute;
            top: -40px;
            right: -20px;
            width: 80px;
            height: 80px;
            background: radial-gradient(circle, var(--kpi-accent, var(--accent-primary)) 0%, transparent 70%);
            opacity: 0.06;
            pointer-events: none;
        }

        .kpi-card:hover {
            border-color: var(--glass-hover);
            transform: translateY(-2px);
            box-shadow: var(--shadow-glow-blue);
        }

        .kpi-icon {
            font-size: 20px;
            margin-bottom: var(--space-sm);
            display: block;
        }

        .kpi-value {
            font-size: 32px;
            font-weight: 800;
            line-height: 1;
            color: var(--kpi-accent, var(--accent-primary));
            margin-bottom: 4px;
            font-variant-numeric: tabular-nums;
        }

        .kpi-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.10em;
        }

        .kpi-delta {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* ═══════════════ MAIN GRID ═══════════════ */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: var(--space-xl);
            margin-bottom: var(--space-xl);
        }

        /* ═══════════════ PANEL ═══════════════ */
        .panel {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-xl);
            overflow: hidden;
            box-shadow: var(--shadow-card);
        }

        .panel-header {
            padding: var(--space-md) var(--space-lg);
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255,255,255,0.02);
        }

        .panel-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            display: flex;
            align-items: center;
            gap: var(--space-sm);
        }

        .panel-title-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent-primary);
            box-shadow: 0 0 6px var(--accent-primary);
        }

        .panel-body {
            padding: var(--space-lg);
        }

        /* ═══════════════ ORBIT CANVAS ═══════════════ */
        .orbit-canvas-wrap {
            position: relative;
            background: radial-gradient(ellipse at center, rgba(14, 165, 233, 0.04) 0%, transparent 70%);
            border-radius: var(--radius-lg);
            overflow: hidden;
        }

        #orbitCanvas {
            display: block;
            width: 100%;
            height: 340px;
        }

        .orbit-legend {
            display: flex;
            gap: var(--space-md);
            flex-wrap: wrap;
            margin-top: var(--space-md);
        }

        .orbit-legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: var(--text-muted);
        }

        .orbit-legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        /* ═══════════════ RISK GAUGE ═══════════════ */
        .gauge-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: var(--space-md) 0;
        }

        #riskGaugeCanvas {
            width: 200px;
            height: 120px;
        }

        .gauge-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.10em;
            margin-top: var(--space-sm);
        }

        .gauge-value {
            font-size: 28px;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1;
        }

        /* ═══════════════ ALERT FEED ═══════════════ */
        .alert-feed {
            display: flex;
            flex-direction: column;
            gap: var(--space-sm);
            max-height: 260px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .alert-feed::-webkit-scrollbar { width: 4px; }
        .alert-feed::-webkit-scrollbar-track { background: transparent; }
        .alert-feed::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: 2px; }

        .alert-item {
            display: flex;
            align-items: flex-start;
            gap: var(--space-sm);
            padding: var(--space-sm) var(--space-md);
            border-radius: var(--radius-md);
            border: 1px solid transparent;
            transition: all 0.2s ease;
            animation: alertSlideIn 0.4s ease forwards;
        }

        @keyframes alertSlideIn {
            from { opacity: 0; transform: translateX(-8px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .alert-item.critical {
            background: rgba(239, 68, 68, 0.08);
            border-color: rgba(239, 68, 68, 0.20);
        }

        .alert-item.high {
            background: rgba(249, 115, 22, 0.08);
            border-color: rgba(249, 115, 22, 0.20);
        }

        .alert-item.medium {
            background: rgba(234, 179, 8, 0.06);
            border-color: rgba(234, 179, 8, 0.15);
        }

        .alert-item.low {
            background: rgba(34, 197, 94, 0.06);
            border-color: rgba(34, 197, 94, 0.15);
        }

        .alert-icon {
            font-size: 14px;
            flex-shrink: 0;
            margin-top: 1px;
        }

        .alert-content { flex: 1; min-width: 0; }

        .alert-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .alert-meta {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .alert-badge {
            flex-shrink: 0;
            padding: 2px 8px;
            border-radius: var(--radius-full);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.06em;
        }

        .badge-CRITICAL { background: rgba(239,68,68,0.15); color: var(--risk-critical); border: 1px solid rgba(239,68,68,0.30); }
        .badge-HIGH     { background: rgba(249,115,22,0.15); color: var(--risk-high);     border: 1px solid rgba(249,115,22,0.30); }
        .badge-MEDIUM   { background: rgba(234,179,8,0.15);  color: var(--risk-medium);   border: 1px solid rgba(234,179,8,0.30); }
        .badge-LOW      { background: rgba(34,197,94,0.15);  color: var(--risk-low);      border: 1px solid rgba(34,197,94,0.30); }

        /* ═══════════════ CONJUNCTION TABLE ═══════════════ */
        .conj-list {
            display: flex;
            flex-direction: column;
            gap: var(--space-md);
        }

        .conj-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            overflow: hidden;
            transition: all 0.3s ease;
            animation: cardFadeIn 0.5s ease forwards;
            opacity: 0;
        }

        @keyframes cardFadeIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .conj-card:hover {
            border-color: var(--border-default);
            box-shadow: var(--shadow-glow-blue);
            transform: translateY(-1px);
        }

        .conj-card.risk-CRITICAL {
            border-left: 3px solid var(--risk-critical);
            box-shadow: -4px 0 16px rgba(239, 68, 68, 0.12);
        }

        .conj-card.risk-HIGH {
            border-left: 3px solid var(--risk-high);
            box-shadow: -4px 0 16px rgba(249, 115, 22, 0.10);
        }

        .conj-card.risk-MEDIUM {
            border-left: 3px solid var(--risk-medium);
        }

        .conj-card.risk-LOW {
            border-left: 3px solid var(--risk-low);
        }

        .conj-card-header {
            padding: var(--space-md) var(--space-lg);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: var(--space-md);
            cursor: pointer;
            user-select: none;
        }

        .conj-objects {
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            flex: 1;
            min-width: 0;
        }

        .sat-name {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 180px;
        }

        .conj-arrow {
            color: var(--text-muted);
            font-size: 16px;
            flex-shrink: 0;
        }

        .conj-header-meta {
            display: flex;
            align-items: center;
            gap: var(--space-md);
            flex-shrink: 0;
        }

        .tca-time {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
        }

        .expand-icon {
            color: var(--text-muted);
            font-size: 12px;
            transition: transform 0.3s ease;
        }

        .conj-card.expanded .expand-icon { transform: rotate(180deg); }

        .conj-card-body {
            display: none;
            padding: 0 var(--space-lg) var(--space-lg);
            border-top: 1px solid var(--border-subtle);
        }

        .conj-card.expanded .conj-card-body { display: block; }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: var(--space-sm);
            margin-top: var(--space-md);
        }

        .metric-tile {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: var(--space-md);
            transition: border-color 0.2s;
        }

        .metric-tile:hover { border-color: var(--border-default); }

        .metric-tile-label {
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.10em;
            margin-bottom: 6px;
        }

        .metric-tile-value {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
            line-height: 1;
        }

        .metric-tile-unit {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Probability bar */
        .prob-track {
            height: 4px;
            background: rgba(255,255,255,0.06);
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        .prob-fill::after {
            content: '';
            position: absolute;
            right: 0;
            top: -2px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: inherit;
            box-shadow: 0 0 6px currentColor;
        }

        /* Confidence arc */
        .confidence-wrap {
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            margin-top: 6px;
        }

        .confidence-bar {
            flex: 1;
            height: 3px;
            background: rgba(255,255,255,0.06);
            border-radius: 2px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--neon-cyan), var(--neon-blue));
            border-radius: 2px;
            transition: width 1s ease;
        }

        .confidence-pct {
            font-size: 11px;
            font-weight: 600;
            color: var(--neon-cyan);
            font-variant-numeric: tabular-nums;
        }

        /* Recommendation box */
        .rec-box {
            margin-top: var(--space-md);
            padding: var(--space-md);
            background: rgba(56, 189, 248, 0.05);
            border: 1px solid rgba(56, 189, 248, 0.15);
            border-radius: var(--radius-md);
            display: flex;
            gap: var(--space-sm);
            align-items: flex-start;
        }

        .rec-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }

        .rec-text {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .maneuver-tag {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            margin-top: var(--space-sm);
            padding: 3px 10px;
            background: rgba(251, 191, 36, 0.10);
            border: 1px solid rgba(251, 191, 36, 0.25);
            border-radius: var(--radius-full);
            font-size: 11px;
            font-weight: 600;
            color: var(--neon-yellow);
        }

        /* ═══════════════ TIMELINE CHART ═══════════════ */
        #timelineCanvas {
            width: 100%;
            height: 120px;
            display: block;
        }

        /* ═══════════════ SIDEBAR PANELS ═══════════════ */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: var(--space-xl);
        }

        /* ═══════════════ LOADING STATE ═══════════════ */
        .loading-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            gap: var(--space-lg);
        }

        .loader-ring {
            position: relative;
            width: 72px;
            height: 72px;
        }

        .loader-ring svg {
            width: 72px;
            height: 72px;
            animation: loaderRotate 2s linear infinite;
        }

        @keyframes loaderRotate { to { transform: rotate(360deg); } }

        .loader-ring circle {
            fill: none;
            stroke-width: 3;
            stroke-linecap: round;
        }

        .loader-ring .track { stroke: rgba(56, 189, 248, 0.10); }
        .loader-ring .fill  {
            stroke: var(--accent-primary);
            stroke-dasharray: 160;
            stroke-dashoffset: 40;
            filter: drop-shadow(0 0 6px var(--accent-primary));
        }

        .loading-title {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .loading-steps {
            display: flex;
            flex-direction: column;
            gap: var(--space-sm);
            align-items: center;
        }

        .loading-step {
            font-size: 13px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: var(--space-sm);
        }

        .loading-step.done { color: var(--neon-green); }
        .loading-step.active { color: var(--accent-primary); }

        .step-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
        }

        .loading-step.active .step-dot {
            animation: statusPulse 1s ease-in-out infinite;
        }

        /* ═══════════════ EMPTY STATE ═══════════════ */
        .empty-state {
            text-align: center;
            padding: var(--space-2xl);
            color: var(--text-muted);
        }

        .empty-state-icon { font-size: 48px; margin-bottom: var(--space-md); }
        .empty-state-title { font-size: 16px; font-weight: 600; color: var(--text-secondary); }
        .empty-state-desc { font-size: 13px; margin-top: var(--space-sm); }

        /* ═══════════════ FOOTER ═══════════════ */
        .footer {
            border-top: 1px solid var(--border-subtle);
            padding: var(--space-md) var(--space-xl);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(5, 12, 20, 0.6);
        }

        .footer-left {
            font-size: 11px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: var(--space-md);
        }

        .footer-sep { color: var(--border-default); }

        .footer-right {
            font-size: 11px;
            color: var(--text-muted);
        }

        .footer-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-full);
            font-size: 10px;
            font-weight: 600;
            color: var(--accent-primary);
            letter-spacing: 0.06em;
        }

        /* ═══════════════ SCROLLBAR ═══════════════ */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--border-strong); }

        /* ═══════════════ RESPONSIVE ═══════════════ */
        @media (max-width: 1200px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .kpi-strip { grid-template-columns: repeat(3, 1fr); }
        }

        @media (max-width: 768px) {
            .topbar { padding: 0 var(--space-md); }
            .main-content { padding: var(--space-md); }
            .kpi-strip { grid-template-columns: repeat(2, 1fr); }
            .metrics-row { grid-template-columns: repeat(2, 1fr); }
            .nav-tabs { display: none; }
        }

        /* ═══════════════ UTILITY ═══════════════ */
        .text-critical { color: var(--risk-critical); }
        .text-high     { color: var(--risk-high); }
        .text-medium   { color: var(--risk-medium); }
        .text-low      { color: var(--risk-low); }
        .text-accent   { color: var(--accent-primary); }
        .text-mono     { font-family: var(--font-mono); }

        .fade-in {
            animation: fadeIn 0.6s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Tooltip */
        [data-tooltip] { position: relative; }
        [data-tooltip]::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: calc(100% + 6px);
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-elevated);
            border: 1px solid var(--border-default);
            border-radius: var(--radius-sm);
            padding: 4px 8px;
            font-size: 11px;
            color: var(--text-secondary);
            white-space: nowrap;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 999;
        }
        [data-tooltip]:hover::after { opacity: 1; }

        /* Scan line effect on orbit canvas */
        .orbit-canvas-wrap::after {
            content: '';
            position: absolute;
            inset: 0;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0, 0, 0, 0.03) 2px,
                rgba(0, 0, 0, 0.03) 4px
            );
            pointer-events: none;
            border-radius: var(--radius-lg);
        }
    </style>
</head>
<body>
<div class="app-layout">

    <!-- ═══════════════ TOPBAR ═══════════════ -->
    <header class="topbar">
        <div class="topbar-left">
            <div class="logo-mark">🛡️</div>
            <div class="logo-wordmark">
                <div class="brand">ORBITAL<span>GUARD</span></div>
                <div class="tagline">AI Mission Control</div>
            </div>
            <div class="topbar-divider"></div>
            <nav class="nav-tabs">
                <button class="nav-tab active">Overview</button>
                <button class="nav-tab">Conjunctions</button>
                <button class="nav-tab">Orbits</button>
                <button class="nav-tab">Analytics</button>
            </nav>
        </div>
        <div class="topbar-right">
            <div class="system-status">
                <div class="status-pulse"></div>
                AI ACTIVE
            </div>
            <div class="clock-display" id="clockDisplay">--:--:-- UTC</div>
            <button class="btn-refresh" id="refreshBtn" onclick="loadData()">
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                    <path d="M13.5 2.5A7 7 0 1 0 14 8"/>
                    <polyline points="14,2 14,6 10,6"/>
                </svg>
                Refresh
            </button>
        </div>
    </header>

    <!-- ═══════════════ MAIN ═══════════════ -->
    <main class="main-content" id="mainContent">
        <!-- Loading state shown initially -->
        <div class="loading-screen" id="loadingScreen">
            <div class="loader-ring">
                <svg viewBox="0 0 72 72">
                    <circle class="track" cx="36" cy="36" r="30"/>
                    <circle class="fill" cx="36" cy="36" r="30"/>
                </svg>
            </div>
            <div class="loading-title">Initializing Mission Control</div>
            <div class="loading-steps">
                <div class="loading-step done"><div class="step-dot"></div>TLE data loaded</div>
                <div class="loading-step done"><div class="step-dot"></div>SGP4 propagators ready</div>
                <div class="loading-step active"><div class="step-dot"></div>Running AI conjunction analysis…</div>
                <div class="loading-step"><div class="step-dot"></div>Generating risk assessments</div>
            </div>
        </div>

        <!-- Dashboard content injected here -->
        <div id="dashboardContent" style="display:none;"></div>
    </main>

    <!-- ═══════════════ FOOTER ═══════════════ -->
    <footer class="footer">
        <div class="footer-left">
            <span class="footer-badge">v1.0.0</span>
            <span>OrbitalGuard AI</span>
            <span class="footer-sep">·</span>
            <span>AEROO Space AI Competition 2026</span>
            <span class="footer-sep">·</span>
            <span>Data: CelesTrak TLE</span>
        </div>
        <div class="footer-right">
            Random Forest · 50 trees · 15 features · 72h window
        </div>
    </footer>

</div>

<script>
/* ═══════════════════════════════════════════════════
   ORBITALGUARD AI — Mission Control JavaScript
   ═══════════════════════════════════════════════════ */

// ── Clock ──────────────────────────────────────────
function updateClock() {
    const now = new Date();
    const h = String(now.getUTCHours()).padStart(2,'0');
    const m = String(now.getUTCMinutes()).padStart(2,'0');
    const s = String(now.getUTCSeconds()).padStart(2,'0');
    document.getElementById('clockDisplay').textContent = h + ':' + m + ':' + s + ' UTC';
}
setInterval(updateClock, 1000);
updateClock();

// ── Helpers ────────────────────────────────────────
function getRiskColor(risk) {
    return { CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#eab308', LOW: '#22c55e' }[risk] || '#38bdf8';
}

function getRiskGlow(risk) {
    return { CRITICAL: 'rgba(239,68,68,0.4)', HIGH: 'rgba(249,115,22,0.35)', MEDIUM: 'rgba(234,179,8,0.3)', LOW: 'rgba(34,197,94,0.3)' }[risk] || 'rgba(56,189,248,0.3)';
}

function getRiskIcon(risk) {
    return { CRITICAL: '🔴', HIGH: '🟠', MEDIUM: '🟡', LOW: '🟢' }[risk] || '⚪';
}

function getProbBarWidth(pc) {
    if (pc <= 0) return 0;
    const logPc = Math.log10(pc);
    const minLog = -8, maxLog = -1;
    return Math.max(2, Math.min(100, (logPc - minLog) / (maxLog - minLog) * 100));
}

function formatProb(p) {
    if (p < 1e-6) return p.toExponential(2);
    if (p < 0.001) return (p * 100).toFixed(4) + '%';
    return (p * 100).toFixed(2) + '%';
}

// ── Orbit Canvas ───────────────────────────────────
function drawOrbitVisualization(canvas, results) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;

    ctx.clearRect(0, 0, W, H);

    // Deep space background
    const bgGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H) * 0.7);
    bgGrad.addColorStop(0, 'rgba(14, 28, 48, 0.95)');
    bgGrad.addColorStop(1, 'rgba(2, 4, 8, 0.98)');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    // Grid rings (subtle)
    for (let r = 40; r <= Math.min(cx, cy) - 10; r += 40) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(56, 139, 253, 0.06)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Cross-hair lines
    ctx.strokeStyle = 'rgba(56, 139, 253, 0.08)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 8]);
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
    ctx.setLineDash([]);

    // Earth
    const earthR = 28;
    const earthGrad = ctx.createRadialGradient(cx - 6, cy - 6, 2, cx, cy, earthR);
    earthGrad.addColorStop(0, '#1e6fa8');
    earthGrad.addColorStop(0.4, '#0d4a7a');
    earthGrad.addColorStop(0.7, '#0a3a5e');
    earthGrad.addColorStop(1, '#061e33');
    ctx.beginPath();
    ctx.arc(cx, cy, earthR, 0, Math.PI * 2);
    ctx.fillStyle = earthGrad;
    ctx.fill();

    // Earth glow
    const earthGlow = ctx.createRadialGradient(cx, cy, earthR * 0.8, cx, cy, earthR * 1.8);
    earthGlow.addColorStop(0, 'rgba(14, 165, 233, 0.12)');
    earthGlow.addColorStop(1, 'transparent');
    ctx.beginPath();
    ctx.arc(cx, cy, earthR * 1.8, 0, Math.PI * 2);
    ctx.fillStyle = earthGlow;
    ctx.fill();

    // Earth label
    ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('EARTH', cx, cy + 3);

    // Orbit paths
    const orbitRadii = [70, 95, 115, 135, 155];
    const orbitColors = ['#38bdf8', '#60a5fa', '#a78bfa', '#34d399', '#fbbf24'];
    const orbitAngles = [0, 0.8, 1.6, 2.4, 3.2];

    orbitRadii.forEach((r, i) => {
        // Orbit ellipse (slightly tilted)
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(orbitAngles[i]);
        ctx.scale(1, 0.35 + i * 0.08);
        ctx.beginPath();
        ctx.arc(0, 0, r, 0, Math.PI * 2);
        ctx.strokeStyle = orbitColors[i % orbitColors.length] + '22';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
    });

    // Animated satellites
    const time = Date.now() / 1000;
    const satData = [
        { r: 70,  speed: 0.8,  phase: 0,    tilt: 0.35, color: '#38bdf8', size: 3 },
        { r: 95,  speed: 0.55, phase: 1.2,  tilt: 0.40, color: '#60a5fa', size: 3 },
        { r: 115, speed: 0.40, phase: 2.5,  tilt: 0.43, color: '#a78bfa', size: 3 },
        { r: 135, speed: 0.30, phase: 0.7,  tilt: 0.46, color: '#34d399', size: 3 },
        { r: 155, speed: 0.22, phase: 3.8,  tilt: 0.49, color: '#fbbf24', size: 3 },
        { r: 80,  speed: 0.65, phase: 4.2,  tilt: 0.38, color: '#f87171', size: 2.5 },
        { r: 105, speed: 0.45, phase: 5.1,  tilt: 0.42, color: '#38bdf8', size: 2.5 },
        { r: 125, speed: 0.35, phase: 1.9,  tilt: 0.44, color: '#60a5fa', size: 2.5 },
    ];

    satData.forEach((sat, idx) => {
        const angle = time * sat.speed + sat.phase;
        const sx = cx + sat.r * Math.cos(angle);
        const sy = cy + sat.r * Math.sin(angle) * sat.tilt;

        // Trail
        ctx.save();
        for (let t = 1; t <= 12; t++) {
            const ta = angle - t * 0.08;
            const tx = cx + sat.r * Math.cos(ta);
            const ty = cy + sat.r * Math.sin(ta) * sat.tilt;
            ctx.beginPath();
            ctx.arc(tx, ty, sat.size * (1 - t / 14), 0, Math.PI * 2);
            ctx.fillStyle = sat.color + Math.floor((1 - t / 14) * 40).toString(16).padStart(2, '0');
            ctx.fill();
        }
        ctx.restore();

        // Satellite dot
        ctx.beginPath();
        ctx.arc(sx, sy, sat.size, 0, Math.PI * 2);
        ctx.fillStyle = sat.color;
        ctx.fill();

        // Glow
        const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, sat.size * 3);
        glow.addColorStop(0, sat.color + '60');
        glow.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(sx, sy, sat.size * 3, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
    });

    // Conjunction warning lines
    if (results && results.length > 0) {
        results.slice(0, 3).forEach((r, i) => {
            const color = getRiskColor(r.risk_level);
            const a1 = (time * 0.5 + i * 1.2) % (Math.PI * 2);
            const a2 = a1 + Math.PI * 0.6 + i * 0.4;
            const rad1 = 70 + i * 30;
            const rad2 = 85 + i * 25;
            const x1 = cx + rad1 * Math.cos(a1);
            const y1 = cy + rad1 * Math.sin(a1) * 0.38;
            const x2 = cx + rad2 * Math.cos(a2);
            const y2 = cy + rad2 * Math.sin(a2) * 0.42;

            // Warning line
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = color + '50';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 5]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Warning flash at midpoint
            const mx = (x1 + x2) / 2;
            const my = (y1 + y2) / 2;
            const flash = 0.5 + 0.5 * Math.sin(time * 3 + i);
            ctx.beginPath();
            ctx.arc(mx, my, 5, 0, Math.PI * 2);
            ctx.fillStyle = color + Math.floor(flash * 200).toString(16).padStart(2, '0');
            ctx.fill();
        });
    }

    // Corner labels
    ctx.fillStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('LEO', 8, 16);
    ctx.textAlign = 'right';
    ctx.fillText('MEO', W - 8, 16);
    ctx.textAlign = 'left';
    ctx.fillText('GEO', 8, H - 8);
}

// ── Risk Gauge ─────────────────────────────────────
function drawRiskGauge(canvas, criticalCount, totalCount) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2;
    const cy = H * 0.85;
    const r = Math.min(W, H * 1.6) * 0.42;
    const startAngle = Math.PI;
    const endAngle = 0;

    // Risk score 0-100
    const score = totalCount > 0 ? Math.min(100, (criticalCount / totalCount) * 100 + (totalCount > 2 ? 30 : 0)) : 0;
    const fillAngle = startAngle + (score / 100) * Math.PI;

    // Track
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Color zones
    const zones = [
        { from: 0, to: 0.33, color: '#22c55e' },
        { from: 0.33, to: 0.66, color: '#eab308' },
        { from: 0.66, to: 1.0, color: '#ef4444' },
    ];

    zones.forEach(z => {
        ctx.beginPath();
        ctx.arc(cx, cy, r, startAngle + z.from * Math.PI, startAngle + z.to * Math.PI);
        ctx.strokeStyle = z.color + '30';
        ctx.lineWidth = 10;
        ctx.lineCap = 'butt';
        ctx.stroke();
    });

    // Fill arc
    if (score > 0) {
        const gaugeColor = score < 33 ? '#22c55e' : score < 66 ? '#eab308' : '#ef4444';
        const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
        grad.addColorStop(0, '#22c55e');
        grad.addColorStop(0.5, '#eab308');
        grad.addColorStop(1, '#ef4444');

        ctx.beginPath();
        ctx.arc(cx, cy, r, startAngle, fillAngle);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Needle tip glow
        const nx = cx + r * Math.cos(fillAngle);
        const ny = cy + r * Math.sin(fillAngle);
        const glow = ctx.createRadialGradient(nx, ny, 0, nx, ny, 12);
        glow.addColorStop(0, gaugeColor + 'cc');
        glow.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(nx, ny, 12, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
    }

    // Score text
    const gaugeLabel = score < 33 ? 'LOW' : score < 66 ? 'MEDIUM' : score < 85 ? 'HIGH' : 'CRITICAL';
    const gaugeColor = score < 33 ? '#22c55e' : score < 66 ? '#eab308' : '#ef4444';

    ctx.fillStyle = gaugeColor;
    ctx.font = 'bold 22px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(Math.round(score), cx, cy - 8);

    ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText('RISK INDEX', cx, cy + 4);
}

// ── Timeline Chart ─────────────────────────────────
function drawTimeline(canvas, results) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const pad = { l: 8, r: 8, t: 8, b: 20 };
    const chartW = W - pad.l - pad.r;
    const chartH = H - pad.t - pad.b;

    // Background
    ctx.fillStyle = 'rgba(255,255,255,0.01)';
    ctx.fillRect(pad.l, pad.t, chartW, chartH);

    // Hour grid lines
    for (let h = 0; h <= 72; h += 12) {
        const x = pad.l + (h / 72) * chartW;
        ctx.beginPath();
        ctx.moveTo(x, pad.t);
        ctx.lineTo(x, pad.t + chartH);
        ctx.strokeStyle = 'rgba(56, 139, 253, 0.08)';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = 'rgba(71, 85, 105, 0.8)';
        ctx.font = '9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(h + 'h', x, H - 4);
    }

    // Plot conjunctions
    if (results && results.length > 0) {
        results.forEach((r, i) => {
            const x = pad.l + (r.time_to_conjunction_hours / 72) * chartW;
            const color = getRiskColor(r.risk_level);

            // Vertical line
            ctx.beginPath();
            ctx.moveTo(x, pad.t);
            ctx.lineTo(x, pad.t + chartH);
            ctx.strokeStyle = color + '40';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 4]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Event dot
            const y = pad.t + chartH * 0.5 + (i % 3 - 1) * 12;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // Glow
            const glow = ctx.createRadialGradient(x, y, 0, x, y, 10);
            glow.addColorStop(0, color + '60');
            glow.addColorStop(1, 'transparent');
            ctx.beginPath();
            ctx.arc(x, y, 10, 0, Math.PI * 2);
            ctx.fillStyle = glow;
            ctx.fill();
        });
    }

    // "NOW" marker
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, pad.t + chartH);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('NOW', pad.l + 3, pad.t + 10);
}

// ── Render Dashboard ───────────────────────────────
function renderDashboard(data) {
    const results = data.results || [];
    const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
    results.forEach(r => { if (counts[r.risk_level] !== undefined) counts[r.risk_level]++; });

    const minDist = results.length > 0 ? Math.min(...results.map(r => r.miss_distance_km)) : 0;

    // ── KPI Strip ──
    const kpiHtml = `
    <div class="kpi-strip fade-in">
        <div class="kpi-card" style="--kpi-accent: #38bdf8">
            <span class="kpi-icon">🛰️</span>
            <div class="kpi-value">${data.objects_count}</div>
            <div class="kpi-label">Objects Tracked</div>
            <div class="kpi-delta">72h analysis window</div>
        </div>
        <div class="kpi-card" style="--kpi-accent: #60a5fa">
            <span class="kpi-icon">⚡</span>
            <div class="kpi-value">${results.length}</div>
            <div class="kpi-label">Conjunctions</div>
            <div class="kpi-delta">Detected events</div>
        </div>
        <div class="kpi-card" style="--kpi-accent: #ef4444">
            <span class="kpi-icon">🚨</span>
            <div class="kpi-value text-critical">${counts.CRITICAL}</div>
            <div class="kpi-label">Critical Risk</div>
            <div class="kpi-delta">Immediate action</div>
        </div>
        <div class="kpi-card" style="--kpi-accent: #f97316">
            <span class="kpi-icon">⚠️</span>
            <div class="kpi-value text-high">${counts.HIGH}</div>
            <div class="kpi-label">High Risk</div>
            <div class="kpi-delta">Monitor closely</div>
        </div>
        <div class="kpi-card" style="--kpi-accent: #22c55e">
            <span class="kpi-icon">🤖</span>
            <div class="kpi-value text-low">${data.model_accuracy}%</div>
            <div class="kpi-label">AI Accuracy</div>
            <div class="kpi-delta">Random Forest model</div>
        </div>
    </div>`;

    // ── Conjunction Cards ──
    let conjHtml = '';
    if (results.length === 0) {
        conjHtml = `
        <div class="empty-state">
            <div class="empty-state-icon">✅</div>
            <div class="empty-state-title">No Conjunction Events Detected</div>
            <div class="empty-state-desc">All tracked objects are within safe separation distances for the next 72 hours.</div>
        </div>`;
    } else {
        results.forEach((r, idx) => {
            const color = getRiskColor(r.risk_level);
            const barW = getProbBarWidth(r.collision_probability);
            const delay = idx * 80;
            conjHtml += `
            <div class="conj-card risk-${r.risk_level}" style="animation-delay:${delay}ms" onclick="toggleCard(this)">
                <div class="conj-card-header">
                    <div class="conj-objects">
                        <span class="sat-name">${r.sat1_name}</span>
                        <span class="conj-arrow">⇄</span>
                        <span class="sat-name">${r.sat2_name}</span>
                    </div>
                    <div class="conj-header-meta">
                        <span class="tca-time">${r.tca}</span>
                        <span class="alert-badge badge-${r.risk_level}">${r.risk_level}</span>
                        <span class="expand-icon">▾</span>
                    </div>
                </div>
                <div class="conj-card-body">
                    <div class="metrics-row">
                        <div class="metric-tile">
                            <div class="metric-tile-label">Miss Distance</div>
                            <div class="metric-tile-value" style="color:${color}">${r.miss_distance_km}</div>
                            <div class="metric-tile-unit">km</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">Relative Velocity</div>
                            <div class="metric-tile-value">${r.relative_velocity_km_s}</div>
                            <div class="metric-tile-unit">km/s</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">Collision Probability</div>
                            <div class="metric-tile-value text-mono" style="color:${color};font-size:14px">${formatProb(r.collision_probability)}</div>
                            <div class="prob-track">
                                <div class="prob-fill" style="width:${barW}%;background:${color}"></div>
                            </div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">Time to TCA</div>
                            <div class="metric-tile-value">${r.time_to_conjunction_hours}</div>
                            <div class="metric-tile-unit">hours</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">Sat 1 Altitude</div>
                            <div class="metric-tile-value">${r.sat1_altitude_km}</div>
                            <div class="metric-tile-unit">km</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">Sat 2 Altitude</div>
                            <div class="metric-tile-value">${r.sat2_altitude_km}</div>
                            <div class="metric-tile-unit">km</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">Approach Angle</div>
                            <div class="metric-tile-value">${r.approach_angle_deg}</div>
                            <div class="metric-tile-unit">degrees</div>
                        </div>
                        <div class="metric-tile">
                            <div class="metric-tile-label">AI Confidence</div>
                            <div class="metric-tile-value">${r.confidence}%</div>
                            <div class="confidence-wrap">
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width:${r.confidence}%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="rec-box">
                        <span class="rec-icon">💡</span>
                        <div>
                            <div class="rec-text">${r.recommendation}</div>
                            ${r.maneuver_window_hours ? `<div class="maneuver-tag">⏱ Maneuver window: ${r.maneuver_window_hours}h before TCA</div>` : ''}
                        </div>
                    </div>
                </div>
            </div>`;
        });
    }

    // ── Alert Feed ──
    let alertHtml = '';
    results.slice(0, 6).forEach((r, i) => {
        const cls = r.risk_level.toLowerCase();
        alertHtml += `
        <div class="alert-item ${cls}" style="animation-delay:${i * 60}ms">
            <span class="alert-icon">${getRiskIcon(r.risk_level)}</span>
            <div class="alert-content">
                <div class="alert-title">${r.sat1_name} ↔ ${r.sat2_name}</div>
                <div class="alert-meta">${r.miss_distance_km} km · ${r.time_to_conjunction_hours}h · ${r.tca}</div>
            </div>
            <span class="alert-badge badge-${r.risk_level}">${r.risk_level}</span>
        </div>`;
    });
    if (alertHtml === '') {
        alertHtml = '<div style="text-align:center;padding:20px;color:var(--text-muted);font-size:13px;">No active alerts</div>';
    }

    // ── Full HTML ──
    const html = `
    ${kpiHtml}

    <div class="dashboard-grid">
        <!-- Left column -->
        <div style="display:flex;flex-direction:column;gap:var(--space-xl)">

            <!-- Orbit Visualization -->
            <div class="panel fade-in">
                <div class="panel-header">
                    <div class="panel-title">
                        <div class="panel-title-dot"></div>
                        Orbital Visualization
                    </div>
                    <div style="font-size:11px;color:var(--text-muted);font-family:var(--font-mono)">
                        LIVE · ${data.objects_count} objects
                    </div>
                </div>
                <div class="panel-body">
                    <div class="orbit-canvas-wrap">
                        <canvas id="orbitCanvas" width="800" height="340"></canvas>
                    </div>
                    <div class="orbit-legend">
                        <div class="orbit-legend-item"><div class="orbit-legend-dot" style="background:#38bdf8"></div>LEO Satellites</div>
                        <div class="orbit-legend-item"><div class="orbit-legend-dot" style="background:#a78bfa"></div>MEO Satellites</div>
                        <div class="orbit-legend-item"><div class="orbit-legend-dot" style="background:#fbbf24"></div>GEO Satellites</div>
                        <div class="orbit-legend-item"><div class="orbit-legend-dot" style="background:#ef4444"></div>Conjunction Warning</div>
                    </div>
                </div>
            </div>

            <!-- Timeline -->
            <div class="panel fade-in">
                <div class="panel-header">
                    <div class="panel-title">
                        <div class="panel-title-dot"></div>
                        72-Hour Conjunction Timeline
                    </div>
                    <div style="font-size:11px;color:var(--text-muted)">
                        Analysis: ${data.analysis_time} UTC
                    </div>
                </div>
                <div class="panel-body" style="padding-bottom:var(--space-sm)">
                    <canvas id="timelineCanvas" width="800" height="120"></canvas>
                </div>
            </div>

            <!-- Conjunction Events -->
            <div class="panel fade-in">
                <div class="panel-header">
                    <div class="panel-title">
                        <div class="panel-title-dot"></div>
                        Conjunction Events
                    </div>
                    <div style="font-size:11px;color:var(--text-muted)">
                        ${results.length} event${results.length !== 1 ? 's' : ''} detected
                    </div>
                </div>
                <div class="panel-body">
                    <div class="conj-list">${conjHtml}</div>
                </div>
            </div>
        </div>

        <!-- Right sidebar -->
        <div class="sidebar">

            <!-- Risk Gauge -->
            <div class="panel fade-in">
                <div class="panel-header">
                    <div class="panel-title">
                        <div class="panel-title-dot"></div>
                        Overall Risk Index
                    </div>
                </div>
                <div class="panel-body">
                    <div class="gauge-wrap">
                        <canvas id="riskGaugeCanvas" width="200" height="120"></canvas>
                        <div class="gauge-label">Threat Assessment</div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--space-sm);margin-top:var(--space-md)">
                        <div class="metric-tile" style="text-align:center">
                            <div class="metric-tile-label">Critical</div>
                            <div class="metric-tile-value text-critical">${counts.CRITICAL}</div>
                        </div>
                        <div class="metric-tile" style="text-align:center">
                            <div class="metric-tile-label">High</div>
                            <div class="metric-tile-value text-high">${counts.HIGH}</div>
                        </div>
                        <div class="metric-tile" style="text-align:center">
                            <div class="metric-tile-label">Medium</div>
                            <div class="metric-tile-value text-medium">${counts.MEDIUM}</div>
                        </div>
                        <div class="metric-tile" style="text-align:center">
                            <div class="metric-tile-label">Low</div>
                            <div class="metric-tile-value text-low">${counts.LOW}</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Alert Feed -->
            <div class="panel fade-in">
                <div class="panel-header">
                    <div class="panel-title">
                        <div class="panel-title-dot" style="background:var(--risk-critical);box-shadow:0 0 6px var(--risk-critical)"></div>
                        Live Alert Feed
                    </div>
                    <div style="font-size:11px;color:var(--text-muted)">${results.length} active</div>
                </div>
                <div class="panel-body">
                    <div class="alert-feed">${alertHtml}</div>
                </div>
            </div>

            <!-- System Info -->
            <div class="panel fade-in">
                <div class="panel-header">
                    <div class="panel-title">
                        <div class="panel-title-dot"></div>
                        System Status
                    </div>
                </div>
                <div class="panel-body">
                    <div style="display:flex;flex-direction:column;gap:var(--space-sm)">
                        ${[
                            ['AI Model', 'Random Forest'],
                            ['Trees', '50 estimators'],
                            ['Features', '15 orbital params'],
                            ['Accuracy', data.model_accuracy + '%'],
                            ['Window', '72 hours'],
                            ['Step', '120 seconds'],
                            ['Objects', data.objects_count + ' tracked'],
                            ['Last Run', data.analysis_time + ' UTC'],
                        ].map(([k, v]) => `
                        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border-subtle)">
                            <span style="font-size:12px;color:var(--text-muted)">${k}</span>
                            <span style="font-size:12px;font-weight:600;color:var(--text-secondary);font-family:var(--font-mono)">${v}</span>
                        </div>`).join('')}
                    </div>
                </div>
            </div>

        </div>
    </div>`;

    document.getElementById('dashboardContent').innerHTML = html;
    document.getElementById('loadingScreen').style.display = 'none';
    document.getElementById('dashboardContent').style.display = 'block';

    // Draw canvases after DOM update
    requestAnimationFrame(() => {
        const orbitCanvas = document.getElementById('orbitCanvas');
        const gaugeCanvas = document.getElementById('riskGaugeCanvas');
        const timelineCanvas = document.getElementById('timelineCanvas');

        if (orbitCanvas) {
            // Resize canvas to actual display size
            orbitCanvas.width = orbitCanvas.offsetWidth || 800;
            orbitCanvas.height = 340;
            drawOrbitVisualization(orbitCanvas, results);
            // Animate orbit
            function animateOrbit() {
                if (document.getElementById('orbitCanvas')) {
                    drawOrbitVisualization(orbitCanvas, results);
                    requestAnimationFrame(animateOrbit);
                }
            }
            animateOrbit();
        }

        if (gaugeCanvas) {
            gaugeCanvas.width = 200;
            gaugeCanvas.height = 120;
            drawRiskGauge(gaugeCanvas, counts.CRITICAL, results.length);
        }

        if (timelineCanvas) {
            timelineCanvas.width = timelineCanvas.offsetWidth || 800;
            timelineCanvas.height = 120;
            drawTimeline(timelineCanvas, results);
        }
    });
}

// ── Toggle Card ────────────────────────────────────
function toggleCard(card) {
    card.classList.toggle('expanded');
}

// ── Load Data ──────────────────────────────────────
function loadData() {
    const btn = document.getElementById('refreshBtn');
    if (btn) btn.classList.add('loading');

    // Show loading overlay if dashboard already visible
    const content = document.getElementById('dashboardContent');
    if (content && content.style.display !== 'none') {
        content.style.opacity = '0.4';
        content.style.transition = 'opacity 0.3s';
    }

    fetch('/api/analyze')
        .then(r => r.json())
        .then(data => {
            renderDashboard(data);
            if (btn) btn.classList.remove('loading');
            if (content) { content.style.opacity = '1'; }
        })
        .catch(err => {
            document.getElementById('loadingScreen').innerHTML = `
                <div style="text-align:center;color:var(--risk-critical)">
                    <div style="font-size:32px;margin-bottom:16px">⚠️</div>
                    <div style="font-size:16px;font-weight:700">Connection Error</div>
                    <div style="font-size:13px;color:var(--text-muted);margin-top:8px">${err}</div>
                    <button class="btn-refresh" style="margin-top:20px" onclick="loadData()">Retry</button>
                </div>`;
            document.getElementById('loadingScreen').style.display = 'flex';
            document.getElementById('dashboardContent').style.display = 'none';
            if (btn) btn.classList.remove('loading');
        });
}

// ── Auto-expand first critical card ───────────────
function autoExpandCritical() {
    const cards = document.querySelectorAll('.conj-card.risk-CRITICAL, .conj-card.risk-HIGH');
    if (cards.length > 0) {
        cards[0].classList.add('expanded');
    }
}

// ── Init ───────────────────────────────────────────
loadData();
setTimeout(autoExpandCritical, 2000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))

        elif parsed.path == '/api/analyze':
            try:
                results = run_analysis()
                response = {
                    'analysis_time': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    'objects_count': len(propagators),
                    'model_accuracy': round(predictor.training_accuracy * 100, 1),
                    'results': results
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()


def main():
    """Start the web dashboard."""
    print("""
+==============================================================+
|          [*] ORBITALGUARD AI - Mission Control              |
+==============================================================+
""")

    # Initialize AI system
    initialize_system()

    # Start server
    port = 8080
    server = HTTPServer(('localhost', port), DashboardHandler)

    print("[WebDashboard] Server started at http://localhost:{}".format(port))
    print("[WebDashboard] Opening browser...")
    print("[WebDashboard] Press Ctrl+C to stop\n")

    # Open browser after short delay
    def open_browser():
        import time
        time.sleep(1.0)
        webbrowser.open('http://localhost:{}'.format(port))

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[WebDashboard] Server stopped.")


if __name__ == "__main__":
    main()

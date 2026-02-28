"""
OrbitalGuard AI - Alert System Module
Generates structured alerts and reports for conjunction events.
"""

import datetime
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Alert:
    """Structured alert for a conjunction event."""
    alert_id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    timestamp: datetime.datetime
    sat1_name: str
    sat2_name: str
    miss_distance_km: float
    collision_probability: float
    time_to_conjunction_hours: float
    recommendation: str
    maneuver_window_hours: Optional[float]
    details: Dict


class AlertSystem:
    """
    Generates and manages collision risk alerts.
    
    In production, this would integrate with:
    - Email/SMS notification services
    - Slack/Teams webhooks
    - REST API endpoints
    - Space-Track.org CDM feeds
    """
    
    SEVERITY_COLORS = {
        'CRITICAL': '[!!!]',
        'HIGH':     '[!! ]',
        'MEDIUM':   '[!  ]',
        'LOW':      '[   ]'
    }
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_counter = 0
    
    def generate_alert(self, conjunction_event, prediction) -> Alert:
        """
        Generate an alert from a conjunction event and ML prediction.
        
        Args:
            conjunction_event: ConjunctionEvent object
            prediction: ModelPrediction object
            
        Returns:
            Alert object
        """
        self.alert_counter += 1
        alert_id = f"OG-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{self.alert_counter:04d}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=prediction.risk_level,
            timestamp=datetime.datetime.utcnow(),
            sat1_name=conjunction_event.sat1_name,
            sat2_name=conjunction_event.sat2_name,
            miss_distance_km=conjunction_event.miss_distance_km,
            collision_probability=prediction.collision_probability,
            time_to_conjunction_hours=conjunction_event.time_to_conjunction_hours,
            recommendation=prediction.recommendation,
            maneuver_window_hours=prediction.maneuver_window_hours,
            details={
                'relative_velocity_km_s': conjunction_event.relative_velocity_km_s,
                'sat1_altitude_km': conjunction_event.sat1_altitude_km,
                'sat2_altitude_km': conjunction_event.sat2_altitude_km,
                'approach_angle_deg': conjunction_event.approach_angle_deg,
                'time_of_closest_approach': conjunction_event.time_of_closest_approach.isoformat(),
                'ml_confidence': prediction.confidence,
                'top_risk_factors': self._get_top_risk_factors(prediction.feature_importance)
            }
        )
        
        self.alerts.append(alert)
        return alert
    
    def _get_top_risk_factors(self, feature_importance: Dict, top_n: int = 3) -> List[str]:
        """Get top contributing risk factors."""
        if not feature_importance:
            return []
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        factor_descriptions = {
            'miss_distance_km': 'Close miss distance',
            'relative_velocity_km_s': 'High relative velocity',
            'approach_angle_deg': 'Head-on approach angle',
            'inverse_distance': 'Extremely close approach',
            'kinetic_energy_proxy': 'High kinetic energy at impact',
            'altitude_difference_km': 'Similar orbital altitude',
            'time_to_conjunction_hours': 'Imminent conjunction'
        }
        
        factors = []
        for feature, importance in sorted_features[:top_n]:
            desc = factor_descriptions.get(feature, feature.replace('_', ' ').title())
            factors.append(f"{desc} ({importance*100:.1f}%)")
        
        return factors
    
    def format_alert(self, alert: Alert) -> str:
        """Format alert as human-readable string."""
        icon = self.SEVERITY_COLORS.get(alert.severity, '⚪')
        
        lines = [
            f"{'='*65}",
            f"{icon} ORBITAL CONJUNCTION ALERT [{alert.severity}]",
            f"{'='*65}",
            f"Alert ID:    {alert.alert_id}",
            f"Generated:   {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"",
            f"OBJECTS INVOLVED:",
            f"  Primary:   {alert.sat1_name}",
            f"  Secondary: {alert.sat2_name}",
            f"",
            f"CONJUNCTION PARAMETERS:",
            f"  Miss Distance:          {alert.miss_distance_km:.3f} km",
            f"  Relative Velocity:      {alert.details['relative_velocity_km_s']:.2f} km/s",
            f"  Approach Angle:         {alert.details['approach_angle_deg']:.1f}°",
            f"  Time to Conjunction:    {alert.time_to_conjunction_hours:.1f} hours",
            f"  Time of Conjunction:    {alert.details['time_of_closest_approach']}",
            f"",
            f"ORBITAL ALTITUDES:",
            f"  {alert.sat1_name}: {alert.details['sat1_altitude_km']:.1f} km",
            f"  {alert.sat2_name}: {alert.details['sat2_altitude_km']:.1f} km",
            f"",
            f"AI RISK ASSESSMENT:",
            f"  Collision Probability:  {alert.collision_probability:.2e} ({alert.collision_probability*100:.4f}%)",
            f"  Risk Level:             {alert.severity}",
            f"  Model Confidence:       {alert.details['ml_confidence']*100:.1f}%",
        ]
        
        if alert.details.get('top_risk_factors'):
            lines.append(f"  Top Risk Factors:")
            for factor in alert.details['top_risk_factors']:
                lines.append(f"    • {factor}")
        
        lines.extend([
            f"",
            f"RECOMMENDATION:",
            f"  {alert.recommendation}",
        ])
        
        if alert.maneuver_window_hours:
            lines.append(f"  Maneuver Window: {alert.maneuver_window_hours:.1f} hours before conjunction")
        
        lines.append(f"{'='*65}")
        
        return '\n'.join(lines)
    
    def generate_summary_report(self, conjunctions: List, predictions: List) -> str:
        """Generate a summary report of all detected conjunctions."""
        
        now = datetime.datetime.utcnow()
        
        # Count by severity
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for pred in predictions:
            level = pred.risk_level if hasattr(pred, 'risk_level') else 'LOW'
            if level in severity_counts:
                severity_counts[level] += 1
        
        lines = [
            f"{'='*65}",
            f"  ORBITALGUARD AI - CONJUNCTION ANALYSIS REPORT",
            f"{'='*65}",
            f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"  Analysis Window: 72 hours",
            f"{'='*65}",
            f"",
            f"EXECUTIVE SUMMARY:",
            f"  Total Conjunctions Detected: {len(conjunctions)}",
            f"  [!!!] CRITICAL: {severity_counts['CRITICAL']}",
            f"  [!! ] HIGH:     {severity_counts['HIGH']}",
            f"  [!  ] MEDIUM:   {severity_counts['MEDIUM']}",
            f"  [   ] LOW:      {severity_counts['LOW']}",
            f"",
        ]
        
        if severity_counts['CRITICAL'] > 0 or severity_counts['HIGH'] > 0:
            lines.extend([
                f"[!] ACTION REQUIRED: {severity_counts['CRITICAL'] + severity_counts['HIGH']} "
                f"high-priority conjunction(s) require immediate attention!",
                f""
            ])
        
        lines.extend([
            f"CONJUNCTION DETAILS:",
            f"{'-'*65}",
        ])
        
        for i, (conj, pred) in enumerate(zip(conjunctions, predictions), 1):
            icon = self.SEVERITY_COLORS.get(pred.risk_level, '[?]')
            lines.extend([
                f"",
                f"#{i} {icon} [{pred.risk_level}] {conj.sat1_name} ↔ {conj.sat2_name}",
                f"   Miss Distance: {conj.miss_distance_km:.3f} km | "
                f"Rel. Velocity: {conj.relative_velocity_km_s:.2f} km/s",
                f"   Collision Probability: {pred.collision_probability:.2e} | "
                f"Time to TCA: {conj.time_to_conjunction_hours:.1f}h",
                f"   Action: {pred.recommendation[:80]}...",
            ])
        
        lines.extend([
            f"",
            f"{'='*65}",
            f"  Powered by OrbitalGuard AI v1.0.0",
            f"  Data sources: CelesTrak TLE, NASA CDM",
            f"  Model: Random Forest Classifier (50 estimators)",
            f"{'='*65}",
        ])
        
        return '\n'.join(lines)
    
    def export_to_json(self, alerts: List[Alert], filepath: str):
        """Export alerts to JSON format for API integration."""
        data = {
            'generated_at': datetime.datetime.utcnow().isoformat(),
            'total_alerts': len(alerts),
            'alerts': []
        }
        
        for alert in alerts:
            data['alerts'].append({
                'alert_id': alert.alert_id,
                'severity': alert.severity,
                'timestamp': alert.timestamp.isoformat(),
                'sat1_name': alert.sat1_name,
                'sat2_name': alert.sat2_name,
                'miss_distance_km': alert.miss_distance_km,
                'collision_probability': alert.collision_probability,
                'time_to_conjunction_hours': alert.time_to_conjunction_hours,
                'recommendation': alert.recommendation,
                'maneuver_window_hours': alert.maneuver_window_hours,
                'details': alert.details
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"[AlertSystem] Exported {len(alerts)} alerts to {filepath}")

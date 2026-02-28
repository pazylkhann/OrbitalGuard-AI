"""
OrbitalGuard AI - Conjunction Detector Module
Detects close approaches (conjunctions) between space objects.
"""

import math
import datetime
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ConjunctionEvent:
    """Represents a close approach event between two space objects."""
    sat1_name: str
    sat2_name: str
    sat1_num: int
    sat2_num: int
    time_of_closest_approach: datetime.datetime
    miss_distance_km: float
    relative_velocity_km_s: float
    sat1_position: np.ndarray
    sat2_position: np.ndarray
    sat1_velocity: np.ndarray
    sat2_velocity: np.ndarray
    
    # Derived features for ML
    sat1_altitude_km: float = 0.0
    sat2_altitude_km: float = 0.0
    approach_angle_deg: float = 0.0
    time_to_conjunction_hours: float = 0.0
    
    # Risk assessment
    risk_level: str = "UNKNOWN"
    collision_probability: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'sat1_name': self.sat1_name,
            'sat2_name': self.sat2_name,
            'sat1_num': self.sat1_num,
            'sat2_num': self.sat2_num,
            'time_of_closest_approach': self.time_of_closest_approach.isoformat(),
            'miss_distance_km': round(self.miss_distance_km, 3),
            'relative_velocity_km_s': round(self.relative_velocity_km_s, 3),
            'sat1_altitude_km': round(self.sat1_altitude_km, 2),
            'sat2_altitude_km': round(self.sat2_altitude_km, 2),
            'approach_angle_deg': round(self.approach_angle_deg, 2),
            'time_to_conjunction_hours': round(self.time_to_conjunction_hours, 2),
            'risk_level': self.risk_level,
            'collision_probability': round(self.collision_probability * 100, 4)
        }


class ConjunctionDetector:
    """
    Detects conjunction events between space objects.
    
    Uses a two-phase approach:
    1. Coarse filter: Check if objects share similar orbital shells
    2. Fine search: Propagate and find minimum distance
    """
    
    # Risk thresholds (km)
    CRITICAL_THRESHOLD = 1.0    # < 1 km: CRITICAL
    HIGH_THRESHOLD = 5.0        # < 5 km: HIGH
    MEDIUM_THRESHOLD = 25.0     # < 25 km: MEDIUM
    LOW_THRESHOLD = 100.0       # < 100 km: LOW (monitor)
    
    RE = 6378.137  # Earth radius km
    
    def __init__(self, propagators: List, search_window_hours: float = 72.0,
                 time_step_seconds: float = 60.0):
        """
        Initialize conjunction detector.
        
        Args:
            propagators: List of SGP4Propagator objects
            search_window_hours: How far ahead to search (default 72h)
            time_step_seconds: Time resolution for propagation (default 60s)
        """
        self.propagators = propagators
        self.search_window_hours = search_window_hours
        self.time_step_seconds = time_step_seconds
    
    def detect_conjunctions(self, start_time: Optional[datetime.datetime] = None,
                           threshold_km: float = 100.0) -> List[ConjunctionEvent]:
        """
        Detect all conjunction events within the search window.
        
        Args:
            start_time: Start of search window (default: now)
            threshold_km: Distance threshold for reporting (km)
            
        Returns:
            List of ConjunctionEvent objects sorted by miss distance
        """
        if start_time is None:
            start_time = datetime.datetime.utcnow()
        
        end_time = start_time + datetime.timedelta(hours=self.search_window_hours)
        
        print(f"[ConjunctionDetector] Scanning {len(self.propagators)} objects")
        print(f"[ConjunctionDetector] Window: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"[ConjunctionDetector] Threshold: {threshold_km} km")
        
        conjunctions = []
        n = len(self.propagators)
        pairs_checked = 0
        
        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                pairs_checked += 1
                event = self._find_closest_approach(
                    self.propagators[i],
                    self.propagators[j],
                    start_time,
                    end_time,
                    threshold_km
                )
                if event is not None:
                    conjunctions.append(event)
        
        print(f"[ConjunctionDetector] Checked {pairs_checked} pairs, found {len(conjunctions)} conjunctions")
        
        # Sort by miss distance (closest first)
        conjunctions.sort(key=lambda x: x.miss_distance_km)
        
        return conjunctions
    
    def _find_closest_approach(self, prop1, prop2, start_time: datetime.datetime,
                               end_time: datetime.datetime, threshold_km: float) -> Optional[ConjunctionEvent]:
        """Find the closest approach between two objects in the time window."""
        
        min_distance = float('inf')
        min_time = None
        min_pos1 = None
        min_pos2 = None
        min_vel1 = None
        min_vel2 = None
        
        current_time = start_time
        dt = datetime.timedelta(seconds=self.time_step_seconds)
        
        # Coarse scan
        while current_time <= end_time:
            try:
                pos1, vel1 = prop1.propagate(current_time)
                pos2, vel2 = prop2.propagate(current_time)
                
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < min_distance:
                    min_distance = distance
                    min_time = current_time
                    min_pos1, min_pos2 = pos1.copy(), pos2.copy()
                    min_vel1, min_vel2 = vel1.copy(), vel2.copy()
                    
            except Exception:
                pass
            
            current_time += dt
        
        # If minimum distance is below threshold, refine with finer search
        if min_distance < threshold_km and min_time is not None:
            # Fine search around the minimum
            fine_start = min_time - datetime.timedelta(seconds=self.time_step_seconds * 2)
            fine_end = min_time + datetime.timedelta(seconds=self.time_step_seconds * 2)
            fine_dt = datetime.timedelta(seconds=10)
            
            current_time = fine_start
            while current_time <= fine_end:
                try:
                    pos1, vel1 = prop1.propagate(current_time)
                    pos2, vel2 = prop2.propagate(current_time)
                    
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        min_time = current_time
                        min_pos1, min_pos2 = pos1.copy(), pos2.copy()
                        min_vel1, min_vel2 = vel1.copy(), vel2.copy()
                        
                except Exception:
                    pass
                
                current_time += fine_dt
            
            # Calculate derived quantities
            rel_vel = np.linalg.norm(min_vel1 - min_vel2)
            
            alt1 = np.linalg.norm(min_pos1) - self.RE
            alt2 = np.linalg.norm(min_pos2) - self.RE
            
            # Approach angle (angle between velocity vectors)
            v1_norm = np.linalg.norm(min_vel1)
            v2_norm = np.linalg.norm(min_vel2)
            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(min_vel1, min_vel2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                approach_angle = math.degrees(math.acos(cos_angle))
            else:
                approach_angle = 0.0
            
            time_to_conj = (min_time - datetime.datetime.utcnow()).total_seconds() / 3600.0
            
            event = ConjunctionEvent(
                sat1_name=prop1.tle['name'],
                sat2_name=prop2.tle['name'],
                sat1_num=prop1.tle['sat_num'],
                sat2_num=prop2.tle['sat_num'],
                time_of_closest_approach=min_time,
                miss_distance_km=min_distance,
                relative_velocity_km_s=rel_vel,
                sat1_position=min_pos1,
                sat2_position=min_pos2,
                sat1_velocity=min_vel1,
                sat2_velocity=min_vel2,
                sat1_altitude_km=max(0, alt1),
                sat2_altitude_km=max(0, alt2),
                approach_angle_deg=approach_angle,
                time_to_conjunction_hours=max(0, time_to_conj)
            )
            
            # Assign preliminary risk level
            event.risk_level = self._assess_risk_level(min_distance)
            
            return event
        
        return None
    
    def _assess_risk_level(self, miss_distance_km: float) -> str:
        """Assign risk level based on miss distance."""
        if miss_distance_km < self.CRITICAL_THRESHOLD:
            return "CRITICAL"
        elif miss_distance_km < self.HIGH_THRESHOLD:
            return "HIGH"
        elif miss_distance_km < self.MEDIUM_THRESHOLD:
            return "MEDIUM"
        elif miss_distance_km < self.LOW_THRESHOLD:
            return "LOW"
        else:
            return "MONITOR"
    
    def extract_features(self, event: ConjunctionEvent) -> np.ndarray:
        """
        Extract ML features from a conjunction event.
        
        Returns feature vector for ML model input.
        """
        features = [
            event.miss_distance_km,
            event.relative_velocity_km_s,
            event.sat1_altitude_km,
            event.sat2_altitude_km,
            abs(event.sat1_altitude_km - event.sat2_altitude_km),  # altitude difference
            event.approach_angle_deg,
            event.time_to_conjunction_hours,
            # Orbital parameters
            math.degrees(event.sat1_velocity[0]) if len(event.sat1_velocity) > 0 else 0,
            np.linalg.norm(event.sat1_velocity),  # sat1 speed
            np.linalg.norm(event.sat2_velocity),  # sat2 speed
            # Relative position components
            abs(event.sat1_position[0] - event.sat2_position[0]),
            abs(event.sat1_position[1] - event.sat2_position[1]),
            abs(event.sat1_position[2] - event.sat2_position[2]),
            # Combined features
            event.miss_distance_km * event.relative_velocity_km_s,  # kinetic energy proxy
            1.0 / (event.miss_distance_km + 0.001),  # inverse distance (risk proxy)
        ]
        
        return np.array(features, dtype=np.float32)


def generate_synthetic_conjunction_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic conjunction data for ML model training.
    
    In production, this would use real NASA Conjunction Data Messages (CDMs).
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (0=safe, 1=collision risk)
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate random conjunction parameters
        miss_distance = np.random.exponential(scale=20.0)  # km
        rel_velocity = np.random.uniform(0.1, 15.0)  # km/s
        alt1 = np.random.uniform(200, 2000)  # km
        alt2 = alt1 + np.random.normal(0, 50)  # similar altitude
        alt_diff = abs(alt1 - alt2)
        approach_angle = np.random.uniform(0, 180)  # degrees
        time_to_conj = np.random.uniform(0, 72)  # hours
        
        # Velocity components
        speed1 = np.random.uniform(6.5, 8.0)  # km/s (LEO range)
        speed2 = np.random.uniform(6.5, 8.0)
        
        # Position differences
        dx = np.random.normal(0, miss_distance)
        dy = np.random.normal(0, miss_distance)
        dz = np.random.normal(0, miss_distance)
        
        features = [
            miss_distance,
            rel_velocity,
            alt1,
            alt2,
            alt_diff,
            approach_angle,
            time_to_conj,
            0.0,  # placeholder
            speed1,
            speed2,
            abs(dx),
            abs(dy),
            abs(dz),
            miss_distance * rel_velocity,
            1.0 / (miss_distance + 0.001)
        ]
        
        # Label: collision risk based on miss distance and velocity
        # High risk: close approach + high relative velocity
        # Using a probabilistic model based on NASA CDM statistics
        if miss_distance < 1.0:
            risk_prob = 0.85
        elif miss_distance < 5.0:
            risk_prob = 0.45 + (rel_velocity / 15.0) * 0.3
        elif miss_distance < 25.0:
            risk_prob = 0.15 + (rel_velocity / 15.0) * 0.1
        else:
            risk_prob = 0.02
        
        label = 1 if np.random.random() < risk_prob else 0
        
        X.append(features)
        y.append(label)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


if __name__ == "__main__":
    print("=== Conjunction Detector Test ===")
    print("Generating synthetic training data...")
    X, y = generate_synthetic_conjunction_data(100)
    print(f"Generated {len(X)} samples, {sum(y)} positive (risk) cases")
    print(f"Feature shape: {X.shape}")
    print(f"Class balance: {sum(y)/len(y)*100:.1f}% risk cases")

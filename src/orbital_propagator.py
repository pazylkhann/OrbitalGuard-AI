"""
OrbitalGuard AI - Orbital Propagator Module
Uses SGP4 algorithm to propagate satellite orbits from TLE data.
"""

import math
import datetime
from typing import Tuple, List, Dict, Optional
import numpy as np


# SGP4 Constants
MU = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
RE = 6378.137     # Earth's radius (km)
J2 = 1.08262998905e-3  # J2 perturbation coefficient
OMEGA_E = 7.2921150e-5  # Earth's rotation rate (rad/s)
KE = 0.0743669161  # sqrt(GM) in units of (ER^3/min^2)^(1/2)
MINUTES_PER_DAY = 1440.0
TWOPI = 2.0 * math.pi


class TLEParser:
    """Parses Two-Line Element (TLE) data for satellites."""
    
    @staticmethod
    def parse_tle(line1: str, line2: str, name: str = "UNKNOWN") -> Dict:
        """
        Parse TLE lines into orbital elements.
        
        Args:
            line1: First TLE line
            line2: Second TLE line
            name: Satellite name
            
        Returns:
            Dictionary with orbital elements
        """
        try:
            # Line 1 parsing
            sat_num = int(line1[2:7])
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            bstar = float(line1[53:54] + '.' + line1[54:59] + 'e' + line1[59:61])
            
            # Line 2 parsing
            inclination = float(line2[8:16])      # degrees
            raan = float(line2[17:25])             # Right Ascension of Ascending Node (degrees)
            eccentricity = float('0.' + line2[26:33])  # eccentricity
            arg_perigee = float(line2[34:42])      # Argument of Perigee (degrees)
            mean_anomaly = float(line2[43:51])     # Mean Anomaly (degrees)
            mean_motion = float(line2[52:63])      # Mean Motion (revs/day)
            
            # Convert epoch
            if epoch_year < 57:
                full_year = 2000 + epoch_year
            else:
                full_year = 1900 + epoch_year
            
            epoch = datetime.datetime(full_year, 1, 1) + datetime.timedelta(days=epoch_day - 1)
            
            return {
                'name': name,
                'sat_num': sat_num,
                'epoch': epoch,
                'inclination': math.radians(inclination),
                'raan': math.radians(raan),
                'eccentricity': eccentricity,
                'arg_perigee': math.radians(arg_perigee),
                'mean_anomaly': math.radians(mean_anomaly),
                'mean_motion': mean_motion * TWOPI / MINUTES_PER_DAY,  # rad/min
                'bstar': bstar,
                'line1': line1,
                'line2': line2
            }
        except Exception as e:
            raise ValueError(f"Failed to parse TLE for {name}: {e}")
    
    @staticmethod
    def parse_tle_file(content: str) -> List[Dict]:
        """Parse multiple TLE entries from a string."""
        satellites = []
        lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
        
        i = 0
        while i < len(lines):
            if lines[i].startswith('1 ') and i + 1 < len(lines) and lines[i+1].startswith('2 '):
                # No name line
                try:
                    sat = TLEParser.parse_tle(lines[i], lines[i+1], f"SAT-{lines[i][2:7].strip()}")
                    satellites.append(sat)
                    i += 2
                except:
                    i += 1
            elif i + 2 < len(lines) and lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
                # Name + two lines
                try:
                    sat = TLEParser.parse_tle(lines[i+1], lines[i+2], lines[i])
                    satellites.append(sat)
                    i += 3
                except:
                    i += 1
            else:
                i += 1
        
        return satellites


class SGP4Propagator:
    """
    Simplified SGP4 orbital propagator.
    Propagates satellite position and velocity from TLE data.
    """
    
    def __init__(self, tle_data: Dict):
        self.tle = tle_data
        self._initialize()
    
    def _initialize(self):
        """Initialize SGP4 constants from TLE data."""
        n0 = self.tle['mean_motion']
        e0 = self.tle['eccentricity']
        i0 = self.tle['inclination']
        
        # Semi-major axis (km)
        a0 = (MU / (n0 * n0 / (60 * 60))) ** (1/3)
        self.semi_major_axis = a0
        
        # Perigee and apogee
        self.perigee = a0 * (1 - e0) - RE
        self.apogee = a0 * (1 + e0) - RE
        
        # Period in minutes
        self.period = TWOPI / n0
    
    def propagate(self, target_time: datetime.datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate satellite to target time.
        
        Args:
            target_time: Target datetime for propagation
            
        Returns:
            Tuple of (position_km, velocity_km_s) as numpy arrays [x, y, z]
        """
        # Time since epoch in minutes
        dt = (target_time - self.tle['epoch']).total_seconds() / 60.0
        
        n = self.tle['mean_motion']
        e = self.tle['eccentricity']
        i = self.tle['inclination']
        omega = self.tle['arg_perigee']
        raan = self.tle['raan']
        M0 = self.tle['mean_anomaly']
        
        # Mean anomaly at target time
        M = M0 + n * dt
        M = M % TWOPI
        
        # Solve Kepler's equation for eccentric anomaly
        E = self._solve_kepler(M, e)
        
        # True anomaly
        nu = 2 * math.atan2(
            math.sqrt(1 + e) * math.sin(E / 2),
            math.sqrt(1 - e) * math.cos(E / 2)
        )
        
        # Semi-major axis
        a = self.semi_major_axis
        
        # Distance from Earth center
        r = a * (1 - e * math.cos(E))
        
        # Position in orbital plane
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)
        
        # Velocity in orbital plane
        h = math.sqrt(MU * a * (1 - e**2))
        vx_orb = -MU / h * math.sin(nu)
        vy_orb = MU / h * (e + math.cos(nu))
        
        # Rotation matrices
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)
        cos_raan = math.cos(raan)
        sin_raan = math.sin(raan)
        cos_i = math.cos(i)
        sin_i = math.sin(i)
        
        # Transform to ECI (Earth-Centered Inertial) frame
        # Rotation matrix R = Rz(-RAAN) * Rx(-i) * Rz(-omega)
        r11 = cos_raan * cos_omega - sin_raan * sin_omega * cos_i
        r12 = -cos_raan * sin_omega - sin_raan * cos_omega * cos_i
        r21 = sin_raan * cos_omega + cos_raan * sin_omega * cos_i
        r22 = -sin_raan * sin_omega + cos_raan * cos_omega * cos_i
        r31 = sin_omega * sin_i
        r32 = cos_omega * sin_i
        
        # Position in ECI
        pos = np.array([
            r11 * x_orb + r12 * y_orb,
            r21 * x_orb + r22 * y_orb,
            r31 * x_orb + r32 * y_orb
        ])
        
        # Velocity in ECI
        vel = np.array([
            r11 * vx_orb + r12 * vy_orb,
            r21 * vx_orb + r22 * vy_orb,
            r31 * vx_orb + r32 * vy_orb
        ]) / 60.0  # Convert from km/min to km/s
        
        return pos, vel
    
    def _solve_kepler(self, M: float, e: float, tol: float = 1e-10) -> float:
        """Solve Kepler's equation M = E - e*sin(E) using Newton-Raphson."""
        E = M  # Initial guess
        for _ in range(100):
            dE = (M - E + e * math.sin(E)) / (1 - e * math.cos(E))
            E += dE
            if abs(dE) < tol:
                break
        return E
    
    def get_orbital_info(self) -> Dict:
        """Get human-readable orbital information."""
        return {
            'name': self.tle['name'],
            'sat_num': self.tle['sat_num'],
            'semi_major_axis_km': round(self.semi_major_axis, 2),
            'perigee_km': round(self.perigee, 2),
            'apogee_km': round(self.apogee, 2),
            'inclination_deg': round(math.degrees(self.tle['inclination']), 4),
            'eccentricity': round(self.tle['eccentricity'], 7),
            'period_min': round(self.period, 2),
            'altitude_type': self._classify_orbit()
        }
    
    def _classify_orbit(self) -> str:
        """Classify orbit type based on altitude."""
        avg_alt = (self.perigee + self.apogee) / 2
        if avg_alt < 2000:
            return "LEO (Low Earth Orbit)"
        elif avg_alt < 35786:
            return "MEO (Medium Earth Orbit)"
        elif 35000 < avg_alt < 36500:
            return "GEO (Geostationary Orbit)"
        else:
            return "HEO (High Earth Orbit)"


def load_sample_tle_data() -> str:
    """
    Returns sample TLE data for demonstration.
    These are real TLE entries from public sources (CelesTrak).
    """
    return """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993
2 25544  51.6400 208.9163 0006317  86.9974 273.1849 15.49815691 00001
STARLINK-1007
1 44713U 19074A   24001.50000000  .00001764  00000-0  13557-3 0  9991
2 44713  53.0539 127.4683 0001350  98.5481 261.5636 15.06391523 00002
NOAA 19
1 33591U 09005A   24001.50000000  .00000065  00000-0  65097-4 0  9994
2 33591  99.1916 109.2345 0013913 312.4567  47.5432 14.12477523 00003
TERRA
1 25994U 99068A   24001.50000000  .00000065  00000-0  65097-4 0  9997
2 25994  98.2000 100.1234 0001234  90.1234 270.0000 14.57116717 00004
COSMOS 2251 DEB
1 33442U 93036SX  24001.50000000  .00001234  00000-0  12345-3 0  9998
2 33442  74.0000 200.0000 0050000 100.0000 260.0000 14.80000000 00005
FENGYUN 1C DEB
1 29228U 99025CCC 24001.50000000  .00002345  00000-0  23456-3 0  9996
2 29228  98.6000 150.0000 0030000 120.0000 240.0000 14.50000000 00006
SENTINEL-2A
1 40697U 15028A   24001.50000000  .00000065  00000-0  65097-4 0  9992
2 40697  98.5700 105.0000 0001000  95.0000 265.0000 14.30000000 00007
LANDSAT 8
1 39084U 13008A   24001.50000000  .00000065  00000-0  65097-4 0  9995
2 39084  98.2200 108.0000 0001500  92.0000 268.0000 14.57000000 00008"""


if __name__ == "__main__":
    print("=== OrbitalGuard AI - Orbital Propagator Test ===\n")
    
    parser = TLEParser()
    tle_content = load_sample_tle_data()
    satellites = parser.parse_tle_file(tle_content)
    
    print(f"Loaded {len(satellites)} satellites\n")
    
    target_time = datetime.datetime.utcnow()
    
    for sat_data in satellites[:3]:
        propagator = SGP4Propagator(sat_data)
        info = propagator.get_orbital_info()
        pos, vel = propagator.propagate(target_time)
        
        print(f"Satellite: {info['name']}")
        print(f"  Orbit Type: {info['altitude_type']}")
        print(f"  Perigee: {info['perigee_km']} km | Apogee: {info['apogee_km']} km")
        print(f"  Inclination: {info['inclination_deg']}°")
        print(f"  Position (ECI): [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] km")
        print(f"  Velocity: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] km/s")
        print()

"""
Geographic Utilities for Fraud Detection
=========================================
Provides distance calculations and impossible travel detection logic.
"""

import math
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

# Approximate coordinates for major cities
# In production, this would use a proper geocoding service
CITY_COORDINATES: Dict[str, Tuple[float, float]] = {
    # UK
    'london': (51.5074, -0.1278),
    'manchester': (53.4808, -2.2426),
    'birmingham': (52.4862, -1.8904),
    'leeds': (53.8008, -1.5491),
    'glasgow': (55.8642, -4.2518),
    'edinburgh': (55.9533, -3.1883),
    'liverpool': (53.4084, -2.9916),
    'bristol': (51.4545, -2.5879),
    'cardiff': (51.4816, -3.1791),
    'belfast': (54.5973, -5.9301),
    
    # Europe
    'paris': (48.8566, 2.3522),
    'berlin': (52.5200, 13.4050),
    'amsterdam': (52.3676, 4.9041),
    'madrid': (40.4168, -3.7038),
    'rome': (41.9028, 12.4964),
    'dublin': (53.3498, -6.2603),
    'brussels': (50.8503, 4.3517),
    'barcelona': (41.3851, 2.1734),
    'vienna': (48.2082, 16.3738),
    'zurich': (47.3769, 8.5417),
    'frankfurt': (50.1109, 8.6821),
    'milan': (45.4642, 9.1900),
    
    # North America
    'new_york': (40.7128, -74.0060),
    'los_angeles': (34.0522, -118.2437),
    'chicago': (41.8781, -87.6298),
    'toronto': (43.6532, -79.3832),
    'san_francisco': (37.7749, -122.4194),
    'miami': (25.7617, -80.1918),
    'boston': (42.3601, -71.0589),
    'seattle': (47.6062, -122.3321),
    'vancouver': (49.2827, -123.1207),
    
    # Asia
    'tokyo': (35.6762, 139.6503),
    'singapore': (1.3521, 103.8198),
    'hong_kong': (22.3193, 114.1694),
    'dubai': (25.2048, 55.2708),
    'mumbai': (19.0760, 72.8777),
    'beijing': (39.9042, 116.4074),
    'shanghai': (31.2304, 121.4737),
    'seoul': (37.5665, 126.9780),
    'bangkok': (13.7563, 100.5018),
    'sydney': (-33.8688, 151.2093),
}

# Country to typical city mapping (for fallback)
COUNTRY_DEFAULT_CITY = {
    'GB': 'london',
    'UK': 'london',
    'US': 'new_york',
    'FR': 'paris',
    'DE': 'berlin',
    'ES': 'madrid',
    'IT': 'rome',
    'JP': 'tokyo',
    'SG': 'singapore',
    'HK': 'hong_kong',
    'AE': 'dubai',
    'IN': 'mumbai',
    'AU': 'sydney',
    'CA': 'toronto',
    'NL': 'amsterdam',
    'IE': 'dublin',
    'BE': 'brussels',
    'CH': 'zurich',
    'AT': 'vienna',
    'KR': 'seoul',
    'TH': 'bangkok',
    'CN': 'beijing',
}

# Maximum physically possible travel speed (commercial aviation)
MAX_POSSIBLE_SPEED_KMH = 1000  # ~620 mph, commercial jet max

# Thresholds for impossible travel detection
IMPOSSIBLE_TRAVEL_THRESHOLDS = {
    'critical': 1500,   # km/h - Definitely impossible (>900 mph)
    'suspicious': 800,  # km/h - Highly suspicious (>500 mph)
    'elevated': 400,    # km/h - Worth flagging (>250 mph, possible but unusual)
}


def haversine_distance(
    lat1: float, 
    lon1: float, 
    lat2: float, 
    lon2: float
) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Uses the Haversine formula for accuracy on a spherical Earth.
    
    Args:
        lat1, lon1: First point (latitude, longitude in degrees)
        lat2, lon2: Second point (latitude, longitude in degrees)
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def get_city_coordinates(city: str, country: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """
    Get coordinates for a city name.
    
    Args:
        city: City name (case-insensitive)
        country: Optional country code for disambiguation
        
    Returns:
        (latitude, longitude) tuple or None if not found
    """
    if not city:
        # Try country default
        if country and country.upper() in COUNTRY_DEFAULT_CITY:
            fallback_city = COUNTRY_DEFAULT_CITY[country.upper()]
            return CITY_COORDINATES.get(fallback_city)
        return None
    
    # Normalize city name
    city_lower = city.lower().strip().replace(' ', '_')
    
    # Direct lookup
    if city_lower in CITY_COORDINATES:
        return CITY_COORDINATES[city_lower]
    
    # Try fuzzy match (simple prefix matching)
    for known_city in CITY_COORDINATES:
        if known_city.startswith(city_lower) or city_lower.startswith(known_city):
            return CITY_COORDINATES[known_city]
    
    # Fallback to country default
    if country and country.upper() in COUNTRY_DEFAULT_CITY:
        fallback_city = COUNTRY_DEFAULT_CITY[country.upper()]
        return CITY_COORDINATES.get(fallback_city)
    
    return None


def calculate_travel_speed(
    city1: str,
    city2: str,
    time_diff_seconds: float,
    country1: Optional[str] = None,
    country2: Optional[str] = None,
) -> Dict[str, any]:
    """
    Calculate travel speed between two locations.
    
    Args:
        city1: Origin city
        city2: Destination city
        time_diff_seconds: Time between transactions in seconds
        country1: Optional country code for city1
        country2: Optional country code for city2
        
    Returns:
        Dict with distance_km, speed_kmh, is_impossible, risk_level
    """
    result = {
        'distance_km': 0.0,
        'speed_kmh': 0.0,
        'is_impossible': False,
        'is_suspicious': False,
        'risk_level': 'normal',
        'coords_found': True,
    }
    
    # Handle same city
    if city1 and city2 and city1.lower() == city2.lower():
        return result
    
    # Get coordinates
    coords1 = get_city_coordinates(city1, country1)
    coords2 = get_city_coordinates(city2, country2)
    
    if not coords1 or not coords2:
        result['coords_found'] = False
        return result
    
    # Calculate distance
    distance = haversine_distance(coords1[0], coords1[1], coords2[0], coords2[1])
    result['distance_km'] = round(distance, 2)
    
    # Calculate speed (avoid division by zero)
    if time_diff_seconds > 0:
        speed_kmh = (distance / time_diff_seconds) * 3600
        result['speed_kmh'] = round(speed_kmh, 2)
        
        # Determine risk level
        if speed_kmh >= IMPOSSIBLE_TRAVEL_THRESHOLDS['critical']:
            result['is_impossible'] = True
            result['is_suspicious'] = True
            result['risk_level'] = 'critical'
        elif speed_kmh >= IMPOSSIBLE_TRAVEL_THRESHOLDS['suspicious']:
            result['is_suspicious'] = True
            result['risk_level'] = 'high'
        elif speed_kmh >= IMPOSSIBLE_TRAVEL_THRESHOLDS['elevated']:
            result['risk_level'] = 'elevated'
    
    return result


def detect_impossible_travel(
    current_city: str,
    current_country: str,
    current_timestamp: datetime,
    last_city: str,
    last_country: str,
    last_timestamp: datetime,
) -> Dict[str, any]:
    """
    Main function to detect impossible travel patterns.
    
    Args:
        current_city: City of current transaction
        current_country: Country of current transaction
        current_timestamp: Timestamp of current transaction
        last_city: City of last transaction
        last_country: Country of last transaction
        last_timestamp: Timestamp of last transaction
        
    Returns:
        Dict with travel analysis results and is_fraud_signal
    """
    # Calculate time difference
    time_diff = (current_timestamp - last_timestamp).total_seconds()
    
    # Skip if negative time (shouldn't happen but safety check)
    if time_diff <= 0:
        return {
            'is_fraud_signal': False,
            'reason': 'Invalid timestamp order',
            'travel_speed_kmh': 0,
            'distance_km': 0,
        }
    
    # Calculate travel metrics
    travel = calculate_travel_speed(
        last_city, current_city, time_diff,
        last_country, current_country
    )
    
    return {
        'is_fraud_signal': travel['is_impossible'] or travel['is_suspicious'],
        'is_impossible_travel': travel['is_impossible'],
        'is_suspicious_travel': travel['is_suspicious'],
        'travel_speed_kmh': travel['speed_kmh'],
        'distance_km': travel['distance_km'],
        'time_diff_minutes': round(time_diff / 60, 1),
        'risk_level': travel['risk_level'],
        'coords_resolved': travel['coords_found'],
    }


# Export feature names for model training
GEO_FEATURE_NAMES = [
    'travel_speed_kmh',
    'travel_distance_km', 
    'is_impossible_travel',
    'is_suspicious_travel',
    'geo_risk_level_encoded',  # 0=normal, 1=elevated, 2=high, 3=critical
]


if __name__ == "__main__":
    # Test the module
    print("Testing geographic utilities...")
    
    # Test distance calculation
    london_coords = CITY_COORDINATES['london']
    paris_coords = CITY_COORDINATES['paris']
    distance = haversine_distance(*london_coords, *paris_coords)
    print(f"London to Paris: {distance:.1f} km (expected ~340 km)")
    
    # Test impossible travel
    now = datetime.now()
    result = detect_impossible_travel(
        current_city="New York",
        current_country="US",
        current_timestamp=now,
        last_city="London",
        last_country="GB",
        last_timestamp=now - timedelta(hours=1),
    )
    print(f"\nLondon -> NYC in 1 hour: {result}")
    
    # Normal travel
    result2 = detect_impossible_travel(
        current_city="Manchester",
        current_country="GB",
        current_timestamp=now,
        last_city="London",
        last_country="GB",
        last_timestamp=now - timedelta(hours=3),
    )
    print(f"London -> Manchester in 3 hours: {result2}")

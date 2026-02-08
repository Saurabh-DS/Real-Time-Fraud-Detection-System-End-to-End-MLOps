"""
Network Analysis Features for Fraud Detection
==============================================
Provides features based on device and IP usage patterns to detect shared resources
and potential fraud rings.
"""

import hashlib
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Risk thresholds for shared resources
SHARED_DEVICE_THRESHOLDS = {
    'critical': 10,  # Device used by 10+ different users
    'high': 5,       # Device used by 5-9 users
    'elevated': 3,   # Device used by 3-4 users
}

IP_RISK_THRESHOLDS = {
    'critical': 20,  # IP used by 20+ different users
    'high': 10,      # IP used by 10-19 users
    'elevated': 5,   # IP used by 5-9 users
}

# Known risky IP patterns (simplified - in prod would use threat intel feeds)
RISKY_IP_PATTERNS = [
    '10.0.0.',      # Private network (shouldn't appear in prod)
    '192.168.',     # Private network
    '127.',         # Localhost
]

# Proxy/VPN service indicators (simplified)
KNOWN_PROXY_ASNS = {
    'AS9009',       # M247 (common VPN provider)
    'AS202425',     # IP Volume Inc
    'AS60068',      # CDN77
    'AS14061',      # DigitalOcean (commonly used for proxies)
}


class DeviceNetworkTracker:
    """
    Tracks device and IP usage patterns in Redis.
    
    Uses Redis sets to efficiently track:
    - Which users have used a specific device
    - Which users have used a specific IP
    - Recent devices/IPs used by a specific user
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl_days = 90  # Keep data for 90 days
        
    def _get_device_key(self, device_fingerprint: str) -> str:
        """Key for tracking users per device."""
        return f"fraud:device_users:{device_fingerprint}"
    
    def _get_ip_key(self, ip_address: str) -> str:
        """Key for tracking users per IP."""
        # Hash IP for privacy in logs
        ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]
        return f"fraud:ip_users:{ip_hash}"
    
    def _get_user_devices_key(self, user_id: str) -> str:
        """Key for tracking devices per user."""
        return f"fraud:user_devices:{user_id}"
    
    def _get_user_ips_key(self, user_id: str) -> str:
        """Key for tracking IPs per user."""
        return f"fraud:user_ips:{user_id}"
    
    def record_transaction(
        self, 
        user_id: str, 
        device_fingerprint: Optional[str], 
        ip_address: Optional[str]
    ) -> Dict[str, any]:
        """
        Record device/IP usage and return risk metrics.
        
        Returns:
            Dict with shared_device_count, shared_ip_count, risk indicators
        """
        result = {
            'device_user_count': 0,
            'ip_user_count': 0,
            'user_device_count': 0,
            'user_ip_count': 0,
            'is_shared_device': False,
            'is_shared_ip': False,
            'device_risk_level': 'normal',
            'ip_risk_level': 'normal',
            'is_new_device': False,
            'is_new_ip': False,
        }
        
        if not self.redis:
            return result
        
        ttl_seconds = self.ttl_days * 24 * 3600
        
        try:
            # Process device fingerprint
            if device_fingerprint:
                device_key = self._get_device_key(device_fingerprint)
                user_devices_key = self._get_user_devices_key(user_id)
                
                # Check if this is a new device for this user
                result['is_new_device'] = not self.redis.sismember(user_devices_key, device_fingerprint)
                
                # Add user to device's user set
                self.redis.sadd(device_key, user_id)
                self.redis.expire(device_key, ttl_seconds)
                
                # Add device to user's device set
                self.redis.sadd(user_devices_key, device_fingerprint)
                self.redis.expire(user_devices_key, ttl_seconds)
                
                # Count users for this device
                result['device_user_count'] = self.redis.scard(device_key)
                result['user_device_count'] = self.redis.scard(user_devices_key)
                
                # Assess device risk
                if result['device_user_count'] >= SHARED_DEVICE_THRESHOLDS['critical']:
                    result['device_risk_level'] = 'critical'
                    result['is_shared_device'] = True
                elif result['device_user_count'] >= SHARED_DEVICE_THRESHOLDS['high']:
                    result['device_risk_level'] = 'high'
                    result['is_shared_device'] = True
                elif result['device_user_count'] >= SHARED_DEVICE_THRESHOLDS['elevated']:
                    result['device_risk_level'] = 'elevated'
                    result['is_shared_device'] = True
            
            # Process IP address
            if ip_address:
                ip_key = self._get_ip_key(ip_address)
                user_ips_key = self._get_user_ips_key(user_id)
                
                # Check if this is a new IP for this user
                result['is_new_ip'] = not self.redis.sismember(user_ips_key, ip_address)
                
                # Add user to IP's user set
                self.redis.sadd(ip_key, user_id)
                self.redis.expire(ip_key, ttl_seconds)
                
                # Add IP to user's IP set
                self.redis.sadd(user_ips_key, ip_address)
                self.redis.expire(user_ips_key, ttl_seconds)
                
                # Count users for this IP
                result['ip_user_count'] = self.redis.scard(ip_key)
                result['user_ip_count'] = self.redis.scard(user_ips_key)
                
                # Assess IP risk
                if result['ip_user_count'] >= IP_RISK_THRESHOLDS['critical']:
                    result['ip_risk_level'] = 'critical'
                    result['is_shared_ip'] = True
                elif result['ip_user_count'] >= IP_RISK_THRESHOLDS['high']:
                    result['ip_risk_level'] = 'high'
                    result['is_shared_ip'] = True
                elif result['ip_user_count'] >= IP_RISK_THRESHOLDS['elevated']:
                    result['ip_risk_level'] = 'elevated'
                    result['is_shared_ip'] = True
                    
        except Exception as e:
            logger.warning(f"Network tracking error: {e}")
        
        return result


def compute_network_features(
    user_id: str,
    device_fingerprint: Optional[str],
    ip_address: Optional[str],
    redis_client,
) -> Dict[str, any]:
    """
    Compute network-based fraud features.
    
    Args:
        user_id: User identifier
        device_fingerprint: Device fingerprint hash (e.g. from browser fingerprinting)
        ip_address: User's IP address
        redis_client: Redis connection
        
    Returns:
        Dict with network features for model inference
    """
    tracker = DeviceNetworkTracker(redis_client)
    
    # Get tracking metrics
    metrics = tracker.record_transaction(user_id, device_fingerprint, ip_address)
    
    # Compute additional features
    features = {
        # Device features
        'shared_device_count': metrics['device_user_count'],
        'is_shared_device': 1 if metrics['is_shared_device'] else 0,
        'is_new_device_for_user': 1 if metrics['is_new_device'] else 0,
        'user_device_diversity': metrics['user_device_count'],
        'device_risk_encoded': _encode_risk_level(metrics['device_risk_level']),
        
        # IP features
        'shared_ip_count': metrics['ip_user_count'],
        'is_shared_ip': 1 if metrics['is_shared_ip'] else 0,
        'is_new_ip_for_user': 1 if metrics['is_new_ip'] else 0,
        'user_ip_diversity': metrics['user_ip_count'],
        'ip_risk_encoded': _encode_risk_level(metrics['ip_risk_level']),
        
        # Combined risk
        'network_risk_score': _calculate_network_risk_score(metrics),
        'is_risky_ip_pattern': _check_risky_ip(ip_address) if ip_address else 0,
    }
    
    return features


def _encode_risk_level(level: str) -> int:
    """Encode risk level as integer for ML."""
    mapping = {
        'normal': 0,
        'elevated': 1,
        'high': 2,
        'critical': 3,
    }
    return mapping.get(level, 0)


def _calculate_network_risk_score(metrics: Dict) -> float:
    """
    Calculate overall network risk score 0-1.
    
    Combines device and IP risk signals.
    """
    score = 0.0
    
    # Device risk contribution (max 0.5)
    if metrics['device_risk_level'] == 'critical':
        score += 0.5
    elif metrics['device_risk_level'] == 'high':
        score += 0.35
    elif metrics['device_risk_level'] == 'elevated':
        score += 0.2
    
    # IP risk contribution (max 0.3)
    if metrics['ip_risk_level'] == 'critical':
        score += 0.3
    elif metrics['ip_risk_level'] == 'high':
        score += 0.2
    elif metrics['ip_risk_level'] == 'elevated':
        score += 0.1
    
    # New device/IP bonus (indicates account takeover risk)
    if metrics['is_new_device']:
        score += 0.1
    if metrics['is_new_ip']:
        score += 0.1
    
    return min(score, 1.0)


def _check_risky_ip(ip_address: str) -> int:
    """Check if IP matches known risky patterns."""
    for pattern in RISKY_IP_PATTERNS:
        if ip_address.startswith(pattern):
            return 1
    return 0


# Export feature names for model training
NETWORK_FEATURE_NAMES = [
    'shared_device_count',
    'is_shared_device',
    'is_new_device_for_user',
    'user_device_diversity',
    'device_risk_encoded',
    'shared_ip_count',
    'is_shared_ip',
    'is_new_ip_for_user',
    'user_ip_diversity',
    'ip_risk_encoded',
    'network_risk_score',
    'is_risky_ip_pattern',
]


if __name__ == "__main__":
    # Test without Redis
    print("Network features module loaded successfully")
    print(f"Feature names: {NETWORK_FEATURE_NAMES}")
    
    # Test risk calculation
    test_metrics = {
        'device_user_count': 7,
        'ip_user_count': 15,
        'device_risk_level': 'high',
        'ip_risk_level': 'high',
        'is_new_device': True,
        'is_new_ip': False,
    }
    score = _calculate_network_risk_score(test_metrics)
    print(f"Test risk score: {score}")

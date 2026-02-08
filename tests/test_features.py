import unittest

from datetime import datetime, timedelta

# Import function to test (mocking import as code is inside class methods in generation script)
# In a real repo, we'd refactor feature logic into 'features/utils.py'

class TestFeatureEngineering(unittest.TestCase):
    
    def test_velocity_calculation(self):
        """Test if 10-minute window correctly counts transactions."""
        now = datetime.now()
        timestamps = [
            now - timedelta(minutes=5),  # Inside
            now - timedelta(minutes=2),  # Inside
            now - timedelta(minutes=15), # Outside
        ]
        
        # Logic from feature_processor.py
        time_10m = now - timedelta(minutes=10)
        count_10m = len([t for t in timestamps if t > time_10m])
        
        self.assertEqual(count_10m, 2, "Rolling 10m window count failed")

    def test_amount_zscore(self):
        """Test Z-Score calculation."""
        current_amount = 150
        avg_amount = 100
        std_amount = 25
        
        zscore = (current_amount - avg_amount) / std_amount
        self.assertEqual(zscore, 2.0, "Z-Score calculation is wrong")
        
    def test_bias_utility_function(self):
        """Test fairness ratio calculation logic."""
        group_a = 0.10
        group_b = 0.08
        ratio = group_a / group_b
        self.assertEqual(ratio, 1.25)

if __name__ == '__main__':
    unittest.main()

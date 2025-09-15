"""
Critical tests for WST transform calculations
Ensures WST features are computed consistently
"""
import pytest
import numpy as np
import torch
from swt_features.wst_transform import WSTTransform, WSTConfig


class TestWSTTransform:
    """Test WST transform calculations"""

    def test_wst_output_shape(self):
        """Test that WST outputs correct shape (128 features)"""
        config = WSTConfig(
            J=2,
            Q=6,
            output_dim=128,
            cache_enabled=False
        )
        
        wst = WSTTransform(config)
        
        # Create 256-bar price window
        prices = np.random.randn(256).astype(np.float32)
        
        # Transform should output 128 features
        features = wst.transform(prices)
        
        assert features.shape == (128,), f"Expected shape (128,), got {features.shape}"
        assert features.dtype == np.float32

    def test_wst_deterministic(self):
        """Test that WST is deterministic for same input"""
        config = WSTConfig(
            J=2,
            Q=6,
            output_dim=128,
            cache_enabled=False
        )
        
        wst = WSTTransform(config)
        
        # Same input should give same output
        prices = np.random.randn(256).astype(np.float32)
        
        features1 = wst.transform(prices)
        features2 = wst.transform(prices)
        
        np.testing.assert_array_almost_equal(features1, features2, decimal=6)

    def test_wst_with_cache(self):
        """Test that caching works correctly"""
        config = WSTConfig(
            J=2,
            Q=6,
            output_dim=128,
            cache_enabled=True
        )
        
        wst = WSTTransform(config)
        
        prices = np.random.randn(256).astype(np.float32)
        cache_key = "test_key_123"
        
        # First call should compute
        features1 = wst.transform(prices, cache_key=cache_key)
        
        # Second call with same key should use cache
        features2 = wst.transform(prices, cache_key=cache_key)
        
        np.testing.assert_array_equal(features1, features2)
        
        # Check cache stats
        stats = wst.get_cache_stats()
        assert stats['cache_hits'] > 0, "Cache should have hits"

    def test_wst_normalization(self):
        """Test that WST features are properly normalized"""
        config = WSTConfig(
            J=2,
            Q=6,
            output_dim=128,
            cache_enabled=False
        )
        
        wst = WSTTransform(config)
        
        # Test with different scale inputs
        small_prices = np.random.randn(256).astype(np.float32) * 0.001
        large_prices = np.random.randn(256).astype(np.float32) * 1000
        
        small_features = wst.transform(small_prices)
        large_features = wst.transform(large_prices)
        
        # Features should be normalized to similar range
        assert np.abs(small_features).max() < 10.0, "Small price features out of range"
        assert np.abs(large_features).max() < 10.0, "Large price features out of range"

    def test_wst_invalid_input(self):
        """Test WST handles invalid inputs gracefully"""
        config = WSTConfig(
            J=2,
            Q=6,
            output_dim=128,
            cache_enabled=False
        )
        
        wst = WSTTransform(config)
        
        # Wrong size input should raise error
        with pytest.raises(Exception):  # FeatureProcessingError
            wrong_size = np.random.randn(100).astype(np.float32)
            wst.transform(wrong_size)
        
        # NaN input should be handled
        nan_prices = np.random.randn(256).astype(np.float32)
        nan_prices[100] = np.nan
        
        # Should either handle NaN or raise clear error
        try:
            features = wst.transform(nan_prices)
            # If it doesn't raise, check no NaNs in output
            assert not np.any(np.isnan(features)), "NaN in WST output"
        except Exception as e:
            # Should be a clear error about NaN
            assert "nan" in str(e).lower() or "invalid" in str(e).lower()

    def test_wst_torch_compatibility(self):
        """Test WST works with PyTorch tensors"""
        config = WSTConfig(
            J=2,
            Q=6,
            output_dim=128,
            cache_enabled=False
        )
        
        wst = WSTTransform(config)
        
        # Test with torch tensor input
        prices_np = np.random.randn(256).astype(np.float32)
        prices_torch = torch.from_numpy(prices_np)
        
        features = wst.transform(prices_torch)
        
        assert features.shape == (128,)
        assert isinstance(features, np.ndarray)
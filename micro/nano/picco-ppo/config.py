"""
Trading configuration for different instruments.
"""

# Instrument-specific configurations
INSTRUMENTS = {
    "GBPJPY": {
        "pip_value": 0.01,      # 1 pip = 0.01 for JPY pairs
        "pip_multiplier": 100,  # To convert price difference to pips
        "spread": 4.0,          # Spread in pips
        "trade_size": 1000.0,   # Standard trade size
    }
}

# Default instrument
DEFAULT_INSTRUMENT = "GBPJPY"

# Get config for current instrument
def get_instrument_config(instrument=DEFAULT_INSTRUMENT):
    """Get configuration for specific instrument."""
    return INSTRUMENTS.get(instrument, INSTRUMENTS[DEFAULT_INSTRUMENT])
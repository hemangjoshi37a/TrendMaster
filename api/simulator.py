import re

class HeadlineSimulator:
    def __init__(self):
        # Bullish keywords & intensifiers
        self.bull_keywords = {
            'launch': 1.5,
            'partnership': 2.0,
            'expansion': 1.8,
            'growth': 1.2,
            'buyback': 2.5,
            'dividend': 1.0,
            'profits': 1.5,
            'breakthrough': 3.0,
            'acquisition': 2.2,
            'upgrade': 1.1,
            'positive': 0.8,
            'surge': 2.0,
            'beat': 1.4,
            'record': 1.6,
            'innovative': 1.2
        }

        # Bearish keywords & intensifiers
        self.bear_keywords = {
            'crash': -4.5,
            'investigation': -3.0,
            'scam': -5.0,
            'lawsuit': -2.5,
            'miss': -1.8,
            'downgrade': -1.5,
            'decline': -1.2,
            'bankruptcy': -7.0,
            'negative': -0.9,
            'plunge': -3.2,
            'drop': -1.4,
            'stagnation': -1.0,
            'fine': -1.2,
            'resignation': -1.5,
            'short': -2.1
        }

        # Intensifiers
        self.intensifiers = {
            'massive': 1.8,
            'huge': 1.5,
            'slight': 0.5,
            'unexpected': 1.4,
            'major': 1.6,
            'minor': 0.7,
            'historic': 2.0,
            'catastrophic': 2.5
        }

    def simulate_shift(self, headline: str):
        headline = headline.lower()
        base_shift = 0.0
        multiplier = 1.0

        # Check for intensifiers
        for word, val in self.intensifiers.items():
            if word in headline:
                multiplier = val
                break

        # Check for bullish/bearish keywords
        found_keywords = []
        for word, val in self.bull_keywords.items():
            if word in headline:
                base_shift += val
                found_keywords.append(word)

        for word, val in self.bear_keywords.items():
            if word in headline:
                base_shift += val
                found_keywords.append(word)

        # Average shift if multiple keywords found (preventing massive stacking)
        if len(found_keywords) > 1:
            base_shift /= len(found_keywords)

        final_shift_pct = base_shift * multiplier
        
        # Max cap to keep simulations realistic
        final_shift_pct = max(min(final_shift_pct, 10.0), -12.0)
        
        return round(final_shift_pct, 2)

    def apply_to_forecast(self, prices: list, shift_pct: float, start_idx: int):
        if shift_pct == 0:
            return prices

        # We only apply shift to the forecast part (from start_idx onwards)
        new_prices = list(prices)
        multiplier = 1.0 + (shift_pct / 100.0)

        # We apply a decaying shift to simulate market pricing in news
        # Day 1 of forecast: 100% of shift
        # Day 10 of forecast: 60% of shift (assuming reversal/correction)
        for i in range(start_idx, len(prices)):
            day_offset = i - start_idx
            decay = 1.0 - (day_offset * 0.04) # SLow decay
            current_multiplier = 1.0 + ((shift_pct / 100.0) * decay)
            new_prices[i] = prices[i] * current_multiplier

        return new_prices

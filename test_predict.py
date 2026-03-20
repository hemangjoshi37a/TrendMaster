import traceback
import __main__
from trendmaster.trendmaster import TransAm, PositionalEncoding

__main__.TransAm = TransAm
__main__.PositionalEncoding = PositionalEncoding

try:
    from api.main import get_real_prediction
    res = get_real_prediction("RELIANCE")
    print(res.get('warning', 'No warning!'))
    print("Preds length:", len(res.get('prices', [])) - res.get('prediction_start_index', 0))
except Exception as e:
    print(traceback.format_exc())

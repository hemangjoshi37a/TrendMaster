from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import asyncio
import os
import re
import time
import torch
import torch.nn as nn
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Import from our library
from trendmaster.trendmaster import TransAm, PositionalEncoding, Inferencer, DataLoader
from api.simulator import HeadlineSimulator

# --- Global State ---
simulator = HeadlineSimulator()
# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# We'll use the 'models' directory created by the TrendMaster class
MODEL_DIR = os.path.join(BASE_DIR, "models")
INSTRUMENTS_FILE = os.path.join(BASE_DIR, "Inference", "all_inst.xlsx")

# --- Global State ---
device = torch.device("cpu")
symbol_to_name = {}
data_loader = DataLoader() # For preprocessing and scaling

# --- Prediction Cache (TTL = 300s) ---
CACHE_FILE = os.path.join(BASE_DIR, "api", "api_cache.json")
CACHE_TTL = 300  # 5 minutes

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(prediction_cache, f)
    except Exception as e:
        print(f"Failed to save cache: {e}")

prediction_cache: Dict[str, dict] = load_cache()  # {"SYMBOL_period": {"data": ..., "timestamp": ...}}

# --- Model Management ---
models: Dict[str, nn.Module] = {}

def get_model(symbol: str):
    symbol = symbol.upper()
    if symbol in models:
        return models[symbol]
    
    # Check possible locations - prioritizing Training/best_model_multi10.pt as it is most likely a state_dict
    paths = [
        os.path.join(MODEL_DIR, f"{symbol}.pt"),
        os.path.join(BASE_DIR, "Training", f"best_model_multi10.pt"),
        os.path.join(BASE_DIR, "Inference", f"best_model_multi10.pt"),
        os.path.join(BASE_DIR, "Inference", f"best_model.pt"),
    ]
    
    for model_path in paths:
        if not os.path.exists(model_path):
            continue
            
        try:
            print(f"Attempting to load model from {model_path}")
            
            # Monkeypatch PyTorch's _LinearWithBias for older torch versions
            import torch.nn.modules.linear
            if not hasattr(torch.nn.modules.linear, '_LinearWithBias'):
                torch.nn.modules.linear._LinearWithBias = torch.nn.modules.linear.Linear
                
            # We need PositionalEncoding and TransAm in __main__
            import __main__
            from trendmaster.trendmaster import TransAm, PositionalEncoding
            __main__.TransAm = TransAm
            __main__.PositionalEncoding = PositionalEncoding
            
            # Load with weights_only=False because whole modules are used
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                # It's a state_dict
                input_size = 6 if "multi" in model_path else 1
                # Standardize on 32 for d_model as it's the library default
                d_model = 32
                if 'input_proj.weight' in checkpoint:
                    d_model = checkpoint['input_proj.weight'].shape[0]
                
                model = TransAm(input_size=input_size, d_model=d_model).to(device)
                model.load_state_dict(checkpoint, strict=False)
                print(f"Successfully loaded state_dict from {model_path}")
            else:
                # It's a whole module
                model = checkpoint
                # Patch missing attributes for older torch versions
                for module in model.modules():
                    if isinstance(module, torch.nn.TransformerEncoderLayer):
                        if not hasattr(module, 'norm_first'):
                            module.norm_first = False
                print(f"Successfully loaded whole module from {model_path}")
                
                # --- Fix for PyTorch version compatibility ---
                for m in model.modules():
                    if isinstance(m, nn.TransformerEncoderLayer):
                        if not hasattr(m, "norm_first"):
                            m.norm_first = False
                        if not hasattr(m, "batch_first"):
                            m.batch_first = False
                # ----------------------------------------------
                
            model.eval()
            models[symbol] = model
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            # Continue to next path
            continue
            
    print(f"No valid model found for {symbol} after checking all paths.")
    return None

def load_instruments():
    global symbol_to_name
    if os.path.exists(INSTRUMENTS_FILE):
        try:
            df = pd.read_excel(INSTRUMENTS_FILE)
            mask = (df['exchange'] == 'NSE') & (df['instrument_type'] == 'EQ')
            nse_df = df[mask]
            for _, row in nse_df.iterrows():
                symbol_to_name[str(row['tradingsymbol']).upper()] = str(row['name'])
            print(f"Loaded {len(symbol_to_name)} NSE instruments")
        except Exception as e:
            print(f"Error loading instruments: {e}")

# --- Lifespan Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_instruments()
    asyncio.create_task(live_market_data_loop())
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prediction Logic ---
def get_real_prediction(symbol, input_window=30, future_steps=10, period="1y", shock_pct=0.0, vix=15.0):
    print(f"--- Starting prediction for {symbol} (period={period}, shock={shock_pct}%) ---")
    yf_symbol = f"{symbol}.NS"
    ticker = yf.Ticker(yf_symbol)
    
    try:
        df = ticker.history(period=period)
    except Exception as e:
        print(f"yfinance error for {symbol}: {e}")
        raise HTTPException(status_code=404, detail=f"Failed to fetch data for {symbol} from Yahoo Finance.")
    
    if df.empty or len(df) < 5:
        print(f"No data found for {symbol}")
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}. It might be delisted or invalid.")

    print(f"Data fetched: {len(df)} rows")

    # Standardize column names for DataLoader.add_technical_indicators
    df_for_indicators = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    
    if shock_pct != 0.0:
        multiplier = 1.0 + (shock_pct / 100.0)
        # Apply shock to the last available day's price
        df_for_indicators.iloc[-1, df_for_indicators.columns.get_loc('close')] *= multiplier
        df_for_indicators.iloc[-1, df_for_indicators.columns.get_loc('high')] *= multiplier
        df_for_indicators.iloc[-1, df_for_indicators.columns.get_loc('low')] *= multiplier
        print(f"Applied shock of {shock_pct}%. New Close: {df_for_indicators['close'].iloc[-1]}")

    if vix > 15.0:
        # Volatility Shock: Add Gaussian noise to the input features
        # The higher the VIX, the more the 'chaos'
        import numpy as np
        chaos_factor = (vix - 15.0) / 200.0  # Normalized factor
        
        # Perturb the last window of data that the Transformer uses
        cols_to_perturb = ['close', 'high', 'low']
        for col in cols_to_perturb:
            if col in df_for_indicators.columns:
                # Calculate standard deviation for scale-aware noise
                std = df_for_indicators[col].rolling(window=10).std().iloc[-1]
                if np.isnan(std): std = df_for_indicators[col].iloc[-1] * 0.01
                
                # Apply noise to the tail
                noise = np.random.normal(0, std * chaos_factor, min(len(df_for_indicators), input_window))
                df_for_indicators.iloc[-len(noise):, df_for_indicators.columns.get_loc(col)] += noise
        
        print(f"Injected chaos factor: {chaos_factor:.3f} (VIX: {vix})")

    df_with_indicators = data_loader.add_technical_indicators(df_for_indicators)
    
    model = get_model(symbol)
    
    # Determine which features were likely used
    input_size = 1
    if model:
        try:
            for name, param in model.named_parameters():
                if 'input_proj.weight' in name:
                    input_size = param.shape[1]
                    break
        except: pass
    
    features = ['close']
    if input_size == 6:
        features = ['close', 'rsi', 'ema_20', 'ema_50', 'macd', 'signal']
    
    print(f"Model features: {features}")

    if not model:
        print(f"No model found for {symbol}, returning historical only")
        close_prices = df['Close'].values[-90:]
        hist_dates = df.index[-90:].strftime('%Y-%m-%d').tolist()
        return {
            "dates": hist_dates,
            "prices": [float(p) for p in close_prices],
            "prediction_start_index": len(close_prices),
            "warning": f"No predictive model found for {symbol}."
        }

    try:
        # CRITICAL: Ensure scaler is fitted before transform in Inferencer
        # We fit on the data we have to avoid NotFittedError
        print("Fitting scaler...")
        data_loader.preprocess_data(df_with_indicators, columns=features, train=True)
        
        print("Running inferencer...")
        inferencer = Inferencer(model, device, data_loader)
        predictions_df = inferencer.predict(symbol, df.index[0], df.index[-1], input_window, future_steps, columns=features, data=df_with_indicators)
        
        print("Prediction successful")
        history_to_show = min(60, len(df))
        # Use df_for_indicators so the shocked point shows up on the chart
        close_prices = df_for_indicators['close'].values[-history_to_show:]
        pred_rescaled = predictions_df['Predicted_Close'].values
        full_series = np.concatenate([close_prices, pred_rescaled])
        
        hist_dates = df.index[-history_to_show:].strftime('%Y-%m-%d').tolist()
        future_dates = predictions_df['Date'].dt.strftime('%Y-%m-%d').tolist()

        # Compute a real confidence score based on prediction smoothness.
        if len(pred_rescaled) > 1:
            daily_returns = np.diff(pred_rescaled) / (np.abs(pred_rescaled[:-1]) + 1e-9)
            cv = np.std(daily_returns) / (np.mean(np.abs(daily_returns)) + 1e-9)
            confidence_score = float(np.clip(100 * (1 / (1 + cv)), 0, 100))
        else:
            confidence_score = 50.0

        # --- Guardian Stop-Loss Calculation ---
        # We use the dynamic historical volatility to set a safe exit
        window = 20
        recent_prices = df_for_indicators['close'].tail(window)
        vol = recent_prices.std()
        if np.isnan(vol) or vol == 0: vol = recent_prices.iloc[-1] * 0.02 # Fallback to 2%
        
        # Adjust multiplier based on horizon (deeper stops for longer terms)
        sl_multiplier = 2.0
        if future_steps >= 60: sl_multiplier = 3.0
        elif future_steps >= 20: sl_multiplier = 2.5
        
        stop_loss = float(recent_prices.iloc[-1] - (sl_multiplier * vol))
        
        return {
            "dates": hist_dates + future_dates,
            "prices": [float(p) for p in full_series],
            "prediction_start_index": history_to_show,
            "confidence_score": round(confidence_score, 1),
            "suggested_stop_loss": round(stop_loss, 2)
        }
    except Exception as e:
        print(f"Prediction error for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        # Return at least historical data so UI doesn't hang
        history_to_show = min(90, len(df))
        close_prices = df_for_indicators['close'].values[-history_to_show:]
        hist_dates = df.index[-history_to_show:].strftime('%Y-%m-%d').tolist()
        return {
            "dates": hist_dates,
            "prices": [float(p) for p in close_prices],
            "prediction_start_index": len(close_prices),
            "warning": f"Model inference failed: {type(e).__name__} - {str(e)[:100]}"
        }

# --- WebSocket & Endpoints ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    async def connect(self, websocket: WebSocket, symbol: str):
        await websocket.accept()
        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
        self.active_connections[symbol].append(websocket)
    def disconnect(self, websocket: WebSocket, symbol: str):
        if symbol in self.active_connections:
            self.active_connections[symbol].remove(websocket)
    async def broadcast(self, symbol: str, message: dict):
        if symbol in self.active_connections:
            for connection in self.active_connections[symbol]:
                try: await connection.send_text(json.dumps(message))
                except: pass

manager = ConnectionManager()

async def live_market_data_loop():
    async def fetch_and_broadcast(symbol: str):
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            price = ticker.fast_info['last_price']
            await manager.broadcast(symbol, {
                "symbol": symbol,
                "price": round(float(price), 2),
                "timestamp": pd.Timestamp.now().isoformat()
            })
        except Exception:
            pass

    while True:
        symbols = list(manager.active_connections.keys())
        if symbols:
            await asyncio.gather(*(fetch_and_broadcast(s) for s in symbols))
        await asyncio.sleep(10)

@app.get("/api/search")
def search_companies(query: str = Query(..., min_length=1)):
    query = query.upper()
    results = []
    for sym, name in symbol_to_name.items():
        if query in sym or query in name.upper():
            results.append({"symbol": sym, "name": name})
            if len(results) >= 10: break
    return results

@app.get("/api/predict")
def predict_stock(
    stock_symbol: str,
    period: str = Query("1y", regex="^(1mo|3mo|6mo|1y|2y|5y|max)$"),
    no_cache: bool = Query(False)
):
    # Input validation
    symbol_upper = stock_symbol.strip().upper()
    if not re.match(r'^[A-Z0-9&_.-]{1,20}$', symbol_upper):
        raise HTTPException(status_code=400, detail="Invalid stock symbol. Use alphanumeric characters only, max 20 chars.")
    
    # Check cache
    cache_key = f"{symbol_upper}_{period}"
    if not no_cache and cache_key in prediction_cache:
        cached = prediction_cache[cache_key]
        if time.time() - cached["timestamp"] < CACHE_TTL:
            print(f"Cache hit for {cache_key}")
            return cached["data"]
    
    company_name = symbol_to_name.get(symbol_upper, symbol_upper)
    
    try:
        data = get_real_prediction(symbol_upper, period=period)
        result = {
            "symbol": symbol_upper,
            "company_name": company_name,
            **data
        }
        # Store in cache
        prediction_cache[cache_key] = {"data": result, "timestamp": time.time()}
        save_cache()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/simulate")
def simulate_shock(
    stock_symbol: str,
    period: str = Query("1y", regex="^(1mo|3mo|6mo|1y|2y|5y|max)$"),
    shock_pct: float = Query(0.0),
    vix: float = Query(15.0)
):
    symbol_upper = stock_symbol.strip().upper()
    if not re.match(r'^[A-Z0-9&_.-]{1,20}$', symbol_upper):
        raise HTTPException(status_code=400, detail="Invalid stock symbol.")
    
    company_name = symbol_to_name.get(symbol_upper, symbol_upper)
    
    try:
        data = get_real_prediction(symbol_upper, period=period, shock_pct=shock_pct, vix=vix)
        return {
            "symbol": symbol_upper,
            "company_name": company_name,
            **data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate-headline")
async def simulate_headline(
    request: dict
):
    """
    Simulate the impact of a custom headline on existing prediction data.
    """
    prediction = request.get("prediction")
    headline = request.get("headline")
    
    if not prediction or not headline:
        raise HTTPException(status_code=400, detail="Missing prediction data or headline.")
    
    try:
        prices = prediction.get("prices", [])
        psi = prediction.get("prediction_start_index", 0)
        
        # Calculate shift
        shift_pct = simulator.simulate_shift(headline)
        
        # Apply to forecast
        new_prices = simulator.apply_to_forecast(prices, shift_pct, psi)
        
        return {
            "symbol": prediction.get("symbol"),
            "company_name": prediction.get("company_name"),
            "dates": prediction.get("dates"),
            "prices": new_prices,
            "prediction_start_index": psi,
            "confidence_score": prediction.get("confidence_score"),
            "simulation_label": f"Shift: {shift_pct:+.2f}%",
            "headline": headline
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

@app.get("/api/multiverse")
def get_multiverse_prediction(
    stock_symbol: str,
    period: str = Query("1y", regex="^(1mo|3mo|6mo|1y|2y|5y|max)$")
):
    symbol_upper = stock_symbol.strip().upper()
    yf_symbol = f"{symbol_upper}.NS"
    ticker = yf.Ticker(yf_symbol)
    
    try:
        df = ticker.history(period=period)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch data for {symbol_upper}")
    
    if df.empty or len(df) < 5:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol_upper}")

    df_for_indicators = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df_with_indicators = data_loader.add_technical_indicators(df_for_indicators)
    model = get_model(symbol_upper)
    
    if not model:
        # Fallback for no model
        close_prices = df['Close'].values[-60:]
        hist_dates = df.index[-60:].strftime('%Y-%m-%d').tolist()
        return {
            "symbol": symbol_upper,
            "dates": hist_dates,
            "prices": [float(p) for p in close_prices],
            "prediction_start_index": len(close_prices),
            "warning": "No predictive model found for multiverse simulation."
        }

    try:
        # Fit scaler
        input_size = 1
        try:
            for name, param in model.named_parameters():
                if 'input_proj.weight' in name:
                    input_size = param.shape[1]
                    break
        except: pass
        
        features = ['close']
        if input_size == 6:
            features = ['close', 'rsi', 'ema_20', 'ema_50', 'macd', 'signal']
            
        data_loader.preprocess_data(df_with_indicators, columns=features, train=True)
        inferencer = Inferencer(model, device, data_loader)
        
        # Define historical data for response
        history_to_show = min(60, len(df))
        close_prices = df_for_indicators['close'].values[-history_to_show:]
        hist_dates = df.index[-history_to_show:].strftime('%Y-%m-%d').tolist()
        
        # Stochastic Prediction (Multiverse)
        mean_df, upper_df, lower_df, all_samples = inferencer.predict(
            symbol_upper, df.index[0], df.index[-1], 30, 10, 
            columns=features, data=df_with_indicators, num_samples=128,
            return_all_samples=True
        )
        
        future_dates = mean_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        
        # Enhanced Risk Analysis
        current_price = float(close_prices[-1])
        risk_stats = inferencer.calculate_risk_metrics(all_samples, current_price)
        
        # Calculate Distribution (Intelligence Gauge)
        final_prices = all_samples[:, -1]
        counts, bins = np.histogram(final_prices, bins=10)
        
        # Chaos Score: standard deviation of final prices relative to mean (mapped 0-10)
        chaos_score = (np.std(final_prices) / (np.mean(final_prices) + 1e-9)) * 100
        
        distribution = {
            "bins": [round(float(b), 2) for b in bins.tolist()],
            "counts": [int(c) for c in counts.tolist()],
            "chaos_score": round(float(chaos_score), 2)
        }
        
        # Scenario Matrix (T+1, T+3, T+5, T+10)
        matrix = []
        for step in [0, 2, 4, 9]: # 0-indexed steps for 1st, 3rd, 5th, 10th day
            if step < len(mean_df):
                matrix.append({
                    "day": step + 1,
                    "date": mean_df['Date'].iloc[step].strftime('%Y-%m-%d'),
                    "mean": float(mean_df['Predicted_Close'].iloc[step]),
                    "upper": float(upper_df['Predicted_Close'].iloc[step]),
                    "lower": float(lower_df['Predicted_Close'].iloc[step])
                })

        return {
            "symbol": symbol_upper,
            "dates": hist_dates + future_dates,
            "prices": [float(p) for p in close_prices] + [float(p) for p in mean_df['Predicted_Close']],
            "cloud_upper": [float(p) for p in upper_df['Predicted_Close'].tolist()],
            "cloud_lower": [float(p) for p in lower_df['Predicted_Close'].tolist()],
            "prediction_start_index": len(close_prices),
            "distribution": distribution,
            "risk_stats": risk_stats,
            "matrix": matrix,
            "is_stochastic": True
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/multiverse/deep-scan")
def historical_stochastic_scan(
    stock_symbol: str, 
    period: str = Query("1y")
):
    """
    Run a heavy historical Monte Carlo scan across the past year.
    """
    symbol_upper = stock_symbol.strip().upper()
    try:
        ticker = yf.Ticker(f"{symbol_upper}.NS")
        df = ticker.history(period=period)
        if df.empty: raise Exception("No data")
        
        # Prepare inferencer
        model = get_model(symbol_upper)
        if not model: raise Exception("No model found")
        
        df_clean = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        data_loader.preprocess_data(df_clean, train=True)
        inferencer = Inferencer(model, device, data_loader)
        
        results = inferencer.stochastic_backtest(symbol_upper, df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d'))
        return {"symbol": symbol_upper, "scan_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-overview")
def market_overview():
    """Fetch real-time market indices."""
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "Bank Nifty": "^NSEBANK"
    }
    result = []
    for name, yf_symbol in indices.items():
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change_pct = ((current - prev) / prev) * 100
                result.append({
                    "name": name,
                    "price": round(float(current), 2),
                    "change_pct": round(float(change_pct), 2)
                })
            elif len(hist) == 1:
                result.append({
                    "name": name,
                    "price": round(float(hist['Close'].iloc[-1]), 2),
                    "change_pct": 0.0
                })
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            result.append({"name": name, "price": 0, "change_pct": 0.0})
    return result

@app.get("/api/backtest")
def backtest_stock(
    stock_symbol: str,
    period: str = Query("2y", regex="^(1y|2y|5y|max)$"),
    test_days: int = Query(90, ge=10, le=365),
    step: int = Query(5, ge=1, le=20)
):
    symbol_upper = stock_symbol.strip().upper()
    yf_symbol = f"{symbol_upper}.NS"
    ticker = yf.Ticker(yf_symbol)
    
    try:
        df = ticker.history(period=period)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch data for {symbol_upper}")
    
    if df.empty or len(df) < test_days + 30:
        raise HTTPException(status_code=400, detail="Insufficient historical data for backtesting.")

    # Prep indicators
    df_for_indicators = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df_with_indicators = data_loader.add_technical_indicators(df_for_indicators)
    
    model = get_model(symbol_upper)
    if not model:
        raise HTTPException(status_code=404, detail=f"No predictive model found for {symbol_upper} to run backtest.")

    # Get input size from model
    input_size = 1
    try:
        for name, param in model.named_parameters():
            if 'input_proj.weight' in name:
                input_size = param.shape[1]
                break
    except: pass
    
    features = ['close']
    if input_size == 6:
        features = ['close', 'rsi', 'ema_20', 'ema_50', 'macd', 'signal']

    inferencer = Inferencer(model, device, data_loader)
    
    # Backtest Window: last `test_days` of available data
    actual_series = df_with_indicators['close'].tolist()
    actual_dates = df_with_indicators.index.strftime('%Y-%m-%d').tolist()
    
    bursts = []
    
    start_idx = len(df_with_indicators) - test_days
    end_idx = len(df_with_indicators) - 10
    
    all_abs_errors = []
    all_sq_errors = []
    direction_hits = 0
    total_direction_tests = 0

    for i in range(start_idx, end_idx, step):
        current_df = df_with_indicators.iloc[:i+1]
        data_loader.preprocess_data(current_df, columns=features, train=True)
        
        try:
            preds_df = inferencer.predict(
                symbol_upper, 
                current_df.index[0], 
                current_df.index[-1], 
                30, 10, 
                columns=features, 
                data=current_df
            )
            
            p_vals = preds_df['Predicted_Close'].tolist()
            p_dates = preds_df['Date'].dt.strftime('%Y-%m-%d').tolist()
            
            actual_slice = df_with_indicators['close'].iloc[i+1 : i+11].tolist()
            if len(actual_slice) > 0:
                for p, a in zip(p_vals, actual_slice):
                    all_abs_errors.append(abs(p - a))
                    all_sq_errors.append((p - a)**2)
                
                if len(actual_slice) >= 5 and len(p_vals) >= 5:
                    actual_dir = actual_slice[4] > actual_series[i]
                    pred_dir = p_vals[4] > actual_series[i]
                    if actual_dir == pred_dir:
                        direction_hits += 1
                    total_direction_tests += 1

                bursts.append({
                    "start_index": i,
                    "dates": p_dates,
                    "prices": p_vals
                })
        except Exception:
            continue

    mae = np.mean(all_abs_errors) if all_abs_errors else 0
    rmse = np.sqrt(np.mean(all_sq_errors)) if all_sq_errors else 0
    win_rate = (direction_hits / total_direction_tests * 100) if total_direction_tests > 0 else 0

    return {
        "symbol": symbol_upper,
        "actual": {
            "dates": actual_dates,
            "prices": actual_series
        },
        "bursts": bursts,
        "metrics": {
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "win_rate": round(float(win_rate), 1)
        }
    }

@app.get("/api/news")
def get_market_news():
    """Fetch global financial news and calculate simulated market impact."""
    indices = ["^NSEI", "^GSPC", "^IXIC", "^FTSE"]
    news_items = []
    
    # Sentiment Keywords
    pos_words = {'rise', 'gain', 'up', 'surge', 'bullish', 'growth', 'beat', 'expansion', 'rate cut', 'easing', 'positive', 'rally', 'profit', 'hit'}
    neg_words = {'fall', 'drop', 'down', 'plunge', 'bearish', 'inflation', 'miss', 'contraction', 'rate hike', 'tightening', 'negative', 'crash', 'loss', 'debt'}

    for idx in indices:
        try:
            ticker = yf.Ticker(idx)
            # yfinance news can be slow, so we take only the latest few
            raw_news = ticker.news[:5]
            for item in raw_news:
                content = item.get('content', {})
                title = content.get('title', 'Market Update')
                summary = content.get('summary', '') or content.get('description', '')
                url = content.get('clickThroughUrl', {}).get('url', '#')
                provider = content.get('provider', {}).get('displayName', 'News')
                
                # Simple Sentiment Analysis
                text = (title + " " + summary).lower()
                score = 0
                for word in pos_words:
                    if word in text: score += 1
                for word in neg_words:
                    if word in text: score -= 1
                
                impact = "NEUTRAL"
                if score > 0: impact = "BULLISH"
                elif score < 0: impact = "BEARISH"
                
                news_items.append({
                    "id": item.get('id'),
                    "title": title,
                    "source": provider,
                    "url": url,
                    "impact": impact,
                    "score": score,
                    "category": "Global" if idx != "^NSEI" else "Indian",
                    "timestamp": pd.Timestamp.now().isoformat() # Ideally we'd parse the real date
                })
        except Exception as e:
            print(f"Error fetching news for {idx}: {e}")
            continue
            
    # Sort by score or just return unique items
    unique_news = {item['id']: item for item in news_items}.values()
    return list(unique_news)[:15]

@app.get("/api/sectors")
def get_sector_heatmap():
    """Fetch real-time daily performance for major Nifty sectors."""
    sector_map = {
        '^NSEBANK': {'name': 'Nifty Bank', 'weight': 35},
        '^CNXIT': {'name': 'Nifty IT', 'weight': 15},
        '^CNXAUTO': {'name': 'Nifty Auto', 'weight': 6},
        '^CNXPHARMA': {'name': 'Nifty Pharma', 'weight': 4},
        '^CNXFMCG': {'name': 'Nifty FMCG', 'weight': 9},
        '^CNXMETAL': {'name': 'Nifty Metal', 'weight': 4},
        '^CNXENERGY': {'name': 'Nifty Energy', 'weight': 12},
        '^CNXREALTY': {'name': 'Nifty Realty', 'weight': 1},
        '^CNXMEDIA': {'name': 'Nifty Media', 'weight': 1}
    }
    
    heatmap_data = []
    for ticker, info in sector_map.items():
        try:
            # yfinance info dictionary often contains 'regularMarketChangePercent'
            # Alternatively, we can download 2 days of history to compute the change
            yticker = yf.Ticker(ticker)
            change = yticker.info.get('regularMarketChangePercent')
            
            # Fallback if 'info' is empty or missing the field
            if change is None:
                hist = yticker.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[0]
                    curr_close = hist['Close'].iloc[1]
                    change = ((curr_close - prev_close) / prev_close) * 100
                else:
                    change = 0.0
                    
            heatmap_data.append({
                'name': info['name'],
                'ticker': ticker,
                'change': round(change, 2),
                'weight': info['weight']
            })
        except Exception as e:
            print(f"Error fetching sector {ticker}: {e}")
            heatmap_data.append({
                'name': info['name'],
                'ticker': ticker,
                'change': 0.0,
                'weight': info['weight']
            })
            
    # Sort by weight (largest blocks first)
    heatmap_data.sort(key=lambda x: x['weight'], reverse=True)
    return heatmap_data

SECTOR_UNIVERSE = {
    'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK', 'PNB', 'IDFCFIRSTB', 'FEDERALBNK'],
    'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'COFORGE', 'PERSISTENT'],
    'Auto': ['TATAMOTORS', 'MARUTI', 'M&M', 'EICHERMOT', 'BAJAJ-AUTO', 'HEROMOTOCO', 'TVSMOTOR'],
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'APOLLOHOSP', 'DIVISLAB', 'ZYDUSLIFE', 'AUROPHARMA'],
    'Energy': ['RELIANCE', 'ONGC', 'BPCL', 'POWERGRID', 'NTPC', 'GAIL', 'ADANIGREEN', 'TATAPOWER'],
    'FMCG': ['ITC', 'HINDUNILVR', 'BRITANNIA', 'NESTLEIND', 'VBL', 'TATACONSUM', 'GODREJCP'],
    'Metal': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'VEDL', 'NMDC', 'SAIL'],
}

@app.get("/api/wealth-advisor")
async def wealth_advisor(
    budget: float = Query(..., gt=0),
    sector: str = Query("All"),
    stock_type: str = Query("Mid", regex="^(Penny|Mid|Large)$"),
    risk: str = Query("Balanced"),
    horizon: int = Query(10)
):
    """
    AI Investment Advisor: Based on budget and preferences, suggests the top 3 stocks.
    """
    import random
    # 1. Determine candidates (Entire sector pool)
    candidate_symbols = []
    if sector == "All":
        for stocks in SECTOR_UNIVERSE.values():
            candidate_symbols.extend(stocks)
    else:
        candidate_symbols = SECTOR_UNIVERSE.get(sector, SECTOR_UNIVERSE['Banking'])

    unique_candidates = list(set(candidate_symbols))
    yf_tickers = [f"{s}.NS" for s in unique_candidates]
    print(f"--- Wealth Advisor Deep Scan for {sector}/{stock_type} (Horizon: {horizon}) ---")
    
    # 2. Robust Price Fetching
    current_prices = {}
    try:
        price_df = yf.download(yf_tickers, period="5d", progress=False)['Close']
        if not price_df.empty:
            for sym in unique_candidates:
                ticker_col = f"{sym}.NS"
                if len(unique_candidates) == 1:
                    val = price_df.dropna()
                    if not val.empty: current_prices[sym] = float(val.iloc[-1])
                elif ticker_col in price_df.columns:
                    val = price_df[ticker_col].dropna()
                    if not val.empty: current_prices[sym] = float(val.iloc[-1])
    except Exception as e:
        print(f"Batch price error: {e}")

    # Fallback to single Ticker fetch for missing prices
    for sym in unique_candidates:
        if sym not in current_prices:
            try:
                p = yf.Ticker(f"{sym}.NS").fast_info['last_price']
                if p: current_prices[sym] = float(p)
            except: pass

    # 3. Comprehensive Scanning (No random.shuffle yet, scan based on price match)
    all_valid_results = []
    
    def check_match(price, s_type, margin=0.0):
        low, high = 0, 1e9
        if s_type == "Penny": low, high = 0, 250 * (1 + margin)
        elif s_type == "Mid": low, high = 250 * (1 - margin), 1500 * (1 + margin)
        elif s_type == "Large": low, high = 1500 * (1 - margin), 1e9
        return low <= price <= high

    # Scan the whole pool for exact price matches first
    matched_syms = [s for s, p in current_prices.items() if check_match(p, stock_type, 0.2)]
    
    if not matched_syms:
        # Emergency relaxation: if nothing in chosen type, just take anything from sector
        matched_syms = list(current_prices.keys())
        print(f"Warning: No exact {stock_type} matches. Falling back to entire sector pool.")

    # Prioritize by predicted alpha (we have to run prediction for all matched syms)
    # To keep it fast, we scan up to 15 candidates
    random.shuffle(matched_syms) 
    candidates_to_analyze = matched_syms[:15]
    
    for sym in candidates_to_analyze:
        try:
            curr_price = current_prices[sym]
            print(f"Analyzing {sym} (₹{curr_price})...")
            
            # Use longer period for advisor to ensure data stability
            pred = get_real_prediction(sym, future_steps=horizon, vix=15.0)
            
            if not pred or "prices" not in pred or "prediction_start_index" not in pred:
                continue
            if "warning" in pred and "No predictive model" in pred["warning"]:
                continue
                
            psi = pred['prediction_start_index']
            prices = pred['prices']
            if len(prices) <= psi: continue
            
            start_price = prices[psi - 1]
            end_price = prices[-1]
            pred_return = ((end_price - start_price) / (start_price + 1e-9)) * 100
            conf = pred.get('confidence_score', 50.0)
            
            all_valid_results.append({
                "symbol": sym,
                "name": symbol_to_name.get(sym, sym),
                "price": round(curr_price, 2),
                "predicted_return": round(pred_return, 2),
                "confidence": conf,
                "wealth_score": pred_return * (conf / 100.0),
                "suggested_stop_loss": pred.get('suggested_stop_loss'),
                "horizon": horizon
            })
        except Exception as e:
            print(f"Advisor scan error for {sym}: {e}")
            continue

    # 4. Final Selection
    all_valid_results.sort(key=lambda x: x['wealth_score'], reverse=True)
    top_3 = all_valid_results[:3]

    if not top_3:
         # Hardest Fallback: If AI fails for all, just return the top 3 by market price alone
         # to avoid the 'No Suitable Stocks Found' loop.
         fallback_candidates = sorted(matched_syms, key=lambda s: current_prices[s], reverse=True)[:3]
         for sym in fallback_candidates:
             top_3.append({
                 "symbol": sym,
                 "name": symbol_to_name.get(sym, sym),
                 "price": round(current_prices[sym], 2),
                 "predicted_return": 0.0,
                 "confidence": 0,
                 "wealth_score": 0,
                 "rationale": "AI Signal unavailable for this symbol. Showing market-cap leader as fallback.",
                 "horizon": horizon
             })

    # Allocation Logic
    splits = [0.45, 0.35, 0.20]
    final_recs = []
    
    for i, res in enumerate(top_3):
        allocated_amt = budget * splits[i]
        shares = int(allocated_amt // res['price'])
        if shares == 0: shares = 1 # Guarantee at least 1
        
        res['recommended_qty'] = shares
        res['total_cost'] = round(shares * res['price'], 2)
        
        # Narrative
        if "rationale" not in res:
            h_label = f"{horizon} days"
            if horizon >= 60: h_label = "3 months"
            elif horizon >= 22: h_label = "1 month"
            
            if res['predicted_return'] > 5:
                res['rationale'] = f"Strong alpha candidate for the {h_label} horizon. Transformer identifies a structural trend with {res['confidence']}% accuracy."
            elif res['predicted_return'] > 0:
                res['rationale'] = f"Steady accumulation recommended. Model projects low-volatility appreciation over the {h_label} period."
            else:
                res['rationale'] = f"Asset preservation play. AI model favors this symbol for stability in the requested {h_label} window."
        
        final_recs.append(res)

    return {
        "budget": budget,
        "horizon": horizon,
        "total_allocated": round(sum(r['total_cost'] for r in final_recs), 2),
        "recommendations": final_recs
    }

    if not final_recs:
        raise HTTPException(status_code=400, detail="Budget too low for horizon-selected leaders.")

    return {
        "budget": budget,
        "horizon": horizon,
        "total_allocated": round(sum(r['total_cost'] for r in final_recs), 2),
        "recommendations": final_recs
    }

@app.get("/api/quote")
def get_live_quote(symbol: str):
    """Fetch real-time snapshot quote for a single symbol."""
    symbol_upper = symbol.strip().upper()
    yf_symbol = f"{symbol_upper}.NS"
    try:
        ticker = yf.Ticker(yf_symbol)
        # fast_info is typically faster for an immediate quote
        price = ticker.fast_info['last_price']
        
        # We can also fetch some basic info to display
        # "name" might be available, fallback to symbol
        name = symbol_to_name.get(symbol_upper, symbol_upper)
        
        return {
            "symbol": symbol_upper,
            "name": name,
            "price": round(float(price), 2),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        print(f"Error fetching quote for {symbol_upper}: {e}")
        raise HTTPException(status_code=404, detail=f"Failed to fetch quote for {symbol_upper}")


@app.websocket("/ws/ticks/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await manager.connect(websocket, symbol.upper())
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol.upper())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

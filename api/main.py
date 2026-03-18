from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import asyncio
import os
import torch
import torch.nn as nn
import yfinance as yf
from typing import List, Dict
from contextlib import asynccontextmanager

# Import from our library
from trendmaster.trendmaster import TransAm, Inferencer, DataLoader

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# We'll use the 'models' directory created by the TrendMaster class
MODEL_DIR = os.path.join(BASE_DIR, "models")
INSTRUMENTS_FILE = os.path.join(BASE_DIR, "Inference", "all_inst.xlsx")

# --- Global State ---
device = torch.device("cpu")
symbol_to_name = {}
data_loader = DataLoader() # For preprocessing and scaling

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
def get_real_prediction(symbol, input_window=30, future_steps=10):
    print(f"--- Starting prediction for {symbol} ---")
    yf_symbol = f"{symbol}.NS"
    ticker = yf.Ticker(yf_symbol)
    
    try:
        df = ticker.history(period="1y")
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
        close_prices = df['Close'].values[-history_to_show:]
        pred_rescaled = predictions_df['Predicted_Close'].values
        full_series = np.concatenate([close_prices, pred_rescaled])
        
        hist_dates = df.index[-history_to_show:].strftime('%Y-%m-%d').tolist()
        future_dates = predictions_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        
        return {
            "dates": hist_dates + future_dates,
            "prices": [float(p) for p in full_series],
            "prediction_start_index": history_to_show
        }
    except Exception as e:
        print(f"Prediction error for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        # Return at least historical data so UI doesn't hang
        history_to_show = min(90, len(df))
        close_prices = df['Close'].values[-history_to_show:]
        hist_dates = df.index[-history_to_show:].strftime('%Y-%m-%d').tolist()
        return {
            "dates": hist_dates,
            "prices": [float(p) for p in close_prices],
            "prediction_start_index": len(close_prices),
            "warning": f"Error during prediction: {str(e)}"
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
    while True:
        for symbol in list(manager.active_connections.keys()):
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                price = ticker.fast_info['last_price']
                await manager.broadcast(symbol, {
                    "symbol": symbol,
                    "price": round(price, 2),
                    "timestamp": pd.Timestamp.now().isoformat()
                })
            except: pass
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
def predict_stock(stock_symbol: str):
    symbol_upper = stock_symbol.upper()
    # If not in our NSE list, try searching yfinance directly as fallback
    company_name = symbol_to_name.get(symbol_upper, symbol_upper)
    
    try:
        data = get_real_prediction(symbol_upper)
        return {
            "symbol": symbol_upper,
            "company_name": company_name,
            **data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

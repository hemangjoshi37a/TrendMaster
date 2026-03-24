import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
import pyotp
import torch
import math

# Must be the very first Streamlit command
st.set_page_config(page_title="TrendMaster Predictions", layout="wide", page_icon="📈")

from trendmaster.trendmaster import (
    DataLoader,
    TransAm,
    Trainer,
    Inferencer,
    set_seed
)

st.title("📈 TrendMaster Predictions Dashboard")

# Session States
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar Authentication and Parameters
with st.sidebar:
    st.header("🔑 Authentication")
    
    if st.session_state.kite is None:
        user_id = st.text_input("Zerodha User ID")
        password = st.text_input("Zerodha Password", type="password")
        totp_key = st.text_input("Zerodha TOTP Key", type="password")
        
        if st.button("Login"):
            if not user_id or not password or not totp_key:
                st.error("Please fill all details.")
            else:
                try:
                    totp = pyotp.TOTP(totp_key)
                    twofa = totp.now()
                    st.session_state.kite = st.session_state.data_loader.authenticate(user_id=user_id, password=password, twofa=twofa)
                    st.success("Successfully Authenticated!")
                except Exception as e:
                    st.error(f"Login Failed: {str(e)}")
    else:
        st.success("Logged in to Zerodha!")

    st.divider()
    st.header("⚙️ Model Parameters")
    symbol = st.text_input("Stock Symbol", "RELIANCE")
    
    col1, col2 = st.columns(2)
    with col1:
        from_date = st.date_input("From Date", datetime.date(2023, 1, 1))
    with col2:
        to_date = st.date_input("To Date", datetime.date(2023, 2, 27))
        
    input_window = st.number_input("Input Window (days)", min_value=1, value=30)
    output_window = st.number_input("Output Window (days)", min_value=1, value=10)
    
    st.subheader("Training Parameters")
    epochs = st.number_input("Training Epochs", min_value=1, max_value=500, value=2)
    batch_size = st.number_input("Batch Size", min_value=1, value=64)
    patience = st.number_input("Early Stopping Patience", min_value=1, value=10)
    
    st.subheader("Inference")
    future_steps = st.number_input("Prediction Future Steps", min_value=1, value=10)


tab1, tab2, tab3 = st.tabs(["🗃️ Data & Training", "🔮 Predictions", "📊 About"])

with tab1:
    st.subheader("Prepare Data & Train Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Prepare Data & Train Model", use_container_width=True):
            if st.session_state.kite is None:
                st.error("Please authenticate in the sidebar first!")
            else:
                with st.spinner("Preparing data and training model..."):
                    # Set seed
                    set_seed(42)
                    
                    try:
                        # Prepare data
                        train_data, test_data = st.session_state.data_loader.prepare_data(
                            symbol=symbol,
                            from_date=from_date.strftime("%Y-%m-%d"),
                            to_date=to_date.strftime("%Y-%m-%d"),
                            input_window=input_window,
                            output_window=output_window,
                            train_test_split=0.8
                        )
                        st.success("Data Prepared Successfully!")
                        
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        st.info(f'Training on {device} device.')
                        
                        model = TransAm(num_layers=2, dropout=0.2).to(device)
                        trainer = Trainer(model, device, learning_rate=0.001)
                        train_losses, val_losses = trainer.train(train_data, test_data, epochs=epochs, batch_size=batch_size, patience=patience)
                        
                        st.session_state.model = model
                        st.session_state.test_data = test_data
                        st.session_state.device = device
                        
                        # Plot losses using Plotly
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))
                        fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss'))
                        fig.update_layout(title="Training vs Validation Loss", xaxis_title="Epochs", yaxis_title="Loss")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success("Training Complete!")
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        
    with col2:
        if st.session_state.model is not None:
            st.success("Model is Trained & Ready in Memory.")
            
            if st.button("Evaluate on Test Data (RMSE/MAE)", use_container_width=True):
                with st.spinner("Evaluating..."):
                    inferencer = Inferencer(st.session_state.model, st.session_state.device, st.session_state.data_loader)
                    
                    # We compute metrics directly here to show in streamlit instead of just printing to console
                    inferencer.model.eval()
                    total_loss, total_mae, total_samples = 0, 0, 0
                    criterion = torch.nn.MSELoss()
                    mae_criterion = torch.nn.L1Loss()
                    test_data = st.session_state.test_data
                    
                    with torch.no_grad():
                        for i in range(0, len(test_data), 32):
                            batch = test_data[i:i+32]
                            import numpy as np
                            input_sequences = np.array([item[0] for item in batch])
                            target_sequences = np.array([item[1] for item in batch])
                            
                            inputs = torch.FloatTensor(input_sequences).unsqueeze(-1).to(st.session_state.device)
                            targets = torch.FloatTensor(target_sequences).unsqueeze(-1).to(st.session_state.device)
                            
                            outputs = inferencer.model(inputs)
                            loss = criterion(outputs, targets)
                            mae_loss = mae_criterion(outputs, targets)
                            
                            batch_size_actual = inputs.size(0)
                            total_loss += loss.item() * batch_size_actual
                            total_mae += mae_loss.item() * batch_size_actual
                            total_samples += batch_size_actual
                    
                    mse = total_loss / total_samples
                    rmse = math.sqrt(mse)
                    mae = total_mae / total_samples
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Test MSE", f"{mse:.6f}")
                    col_m2.metric("Test RMSE", f"{rmse:.6f}")
                    col_m3.metric("Test MAE", f"{mae:.6f}")
                    
with tab2:
    st.subheader("Make Future Predictions")
    
    if st.button("🔮 Predict Future Stock Price", use_container_width=True):
        if st.session_state.kite is None:
             st.error("Please authenticate first!")
        elif st.session_state.model is None:
             st.error("Please train your model in the first tab!")
        else:
             with st.spinner("Generating predictions..."):
                 try:
                     inferencer = Inferencer(st.session_state.model, st.session_state.device, st.session_state.data_loader)
                     predictions_df = inferencer.predict(
                         symbol=symbol,
                         from_date=to_date.strftime("%Y-%m-%d"),
                         to_date=to_date.strftime("%Y-%m-%d"),  # we predict *from* the end date
                         input_window=input_window,
                         future_steps=future_steps
                     )
                     
                     st.write(f"### Prediction Results for {symbol}")
                     
                     # Render interactive chart
                     fig2 = go.Figure()
                     
                     # Check if we should render historical data to add context
                     try:
                         data = st.session_state.data_loader.load_or_download_data(symbol, from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))
                         fig2.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines', name='Historical Close', line=dict(color='blue')))
                     except:
                         pass
                     
                     fig2.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted_Close'], mode='lines+markers', name='Predicted Close', line=dict(color='red', width=3)))
                     fig2.update_layout(title=f"Prediction for {symbol}", xaxis_title="Date", yaxis_title="Price")
                     
                     st.plotly_chart(fig2, use_container_width=True)
                     st.dataframe(predictions_df, use_container_width=True)
                     
                 except Exception as e:
                     st.error(f"Error during prediction: {str(e)}")

with tab3:
    st.markdown("""
    ## About TrendMaster UI
    
    This dashboard interacts directly with the **TrendMaster** Python Deep Learning Transformer architecture to accurately anticipate stock market movements.
    
    - **Advanced Transformer**: Uses Multi-head attention instead of standard LSTMs for superior temporal sequence prediction.
    - **Data Preprocessing**: Ensures zero data leakage with strict MinMax scaler train/test splitting.
    - **Interactive Visualization**: Leverages Streamlit and Plotly to display real-time training evaluation and stunning prediction charts.
    """)

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import socket
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI MQTT ====================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_SENSOR = "iot/streetlight/data"

# ==================== SESSION STATE INIT ====================
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

if "connection_status" not in st.session_state:
    st.session_state.connection_status = "❌ TIDAK TERKONEKSI"

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_data" not in st.session_state:
    st.session_state.last_data = None

if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None

if "ml_predictions" not in st.session_state:
    st.session_state.ml_predictions = {}

# ==================== SIMPLE ML PREDICTION ====================
def simple_ml_prediction(intensity, voltage):
    """Prediksi sederhana berdasarkan rules jika model tidak bekerja"""
    
    # Rule-based prediction (fallback)
    if intensity < 30:
        return "Lampu MENYALA (Gelap)"
    elif intensity > 70:
        return "Lampu MATI (Terang)"
    else:
        return "Kondisi NORMAL (Sedang)"
    
    return "Tidak dapat diprediksi"

# ==================== TRY LOAD ML MODELS ====================
def try_load_models():
    """Coba load model ML dengan error handling"""
    models_info = {}
    
    model_files = {
        'decision_tree': 'decision_tree.pkl',
        'knn': 'k-nearest_neighbors.pkl', 
        'logistic_regression': 'logistic_regression.pkl',
        'scaler': 'feature_scaler.pkl',
        'encoder': 'target_encoder.pkl'
    }
    
    for model_name, filename in model_files.items():
        try:
            model = joblib.load(filename)
            models_info[model_name] = {
                'loaded': True,
                'model': model,
                'filename': filename
            }
            print(f"✅ {model_name} loaded from {filename}")
        except Exception as e:
            models_info[model_name] = {
                'loaded': False,
                'error': str(e),
                'filename': filename
            }
            print(f"❌ {model_name} failed to load: {e}")
    
    return models_info

# Coba load models
ml_models = try_load_models()

# ==================== MQTT CALLBACKS ====================
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        st.session_state.mqtt_connected = True
        st.session_state.connection_status = "✅ TERKONEKSI"
        client.subscribe(MQTT_TOPIC_SENSOR)
        print("✅ Connected to MQTT broker")
    else:
        st.session_state.mqtt_connected = False
        st.session_state.connection_status = f"❌ ERROR (code: {rc})"
        print(f"❌ Connection failed: {rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8', errors='ignore')
        print(f"📥 Received: {payload}")
        
        # Parse data dari ESP32
        if payload.startswith("{") and payload.endswith("}"):
            clean_payload = payload[1:-1]
            parts = clean_payload.split(";")
            
            if len(parts) == 3:
                timestamp_str = parts[0].strip()
                intensity_str = parts[1].strip()
                voltage_str = parts[2].strip()
                
                # Parse values
                try:
                    intensity = float(intensity_str)
                except:
                    intensity = None
                
                try:
                    voltage = float(voltage_str)
                except:
                    voltage = None
                
                # Parse timestamp
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except:
                    timestamp = datetime.now()
                
                # Determine states
                if voltage == 0.0:
                    relay_state = "MATI"
                    lamp_state = "MENYALA"
                elif voltage == 220.0:
                    relay_state = "AKTIF"
                    lamp_state = "MATI"
                else:
                    relay_state = "UNKNOWN"
                    lamp_state = "UNKNOWN"
                
                # Simple ML prediction
                ml_prediction = simple_ml_prediction(intensity, voltage) if intensity is not None else "N/A"
                
                # Create data row
                row = {
                    "timestamp": timestamp,
                    "intensity": intensity,
                    "voltage": voltage,
                    "relay_state": relay_state,
                    "lamp_state": lamp_state,
                    "ml_prediction": ml_prediction,
                    "source": "MQTT REAL"
                }
                
                # Update session state
                st.session_state.last_data = row
                st.session_state.logs.append(row)
                
                # Keep last 500 records
                if len(st.session_state.logs) > 500:
                    st.session_state.logs = st.session_state.logs[-500:]
                
                print(f"✅ Data stored: Intensity={intensity}, ML={ml_prediction}")
                
    except Exception as e:
        print(f"❌ Error processing MQTT: {e}")

# ==================== FUNGSI KONEKSI ====================
def connect_mqtt():
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        st.session_state.mqtt_client = client
        return True
    except Exception as e:
        st.session_state.connection_status = f"❌ ERROR: {str(e)[:50]}"
        return False

# ==================== STREAMLIT UI ====================
st.set_page_config(
    page_title="Smart Streetlight Dashboard",
    page_icon="💡",
    layout="wide"
)

st.title("💡 SMART STREETLIGHT MONITORING")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ KONTROL")
    
    # Connection Status
    st.subheader("🔗 STATUS KONEKSI")
    
    if st.session_state.mqtt_connected:
        st.success("✅ TERHUBUNG")
    else:
        st.error("❌ TIDAK TERHUBUNG")
    
    st.write(f"**Status:** {st.session_state.connection_status}")
    
    # Connection Buttons
    st.markdown("---")
    
    if not st.session_state.mqtt_connected:
        if st.button("🔗 Sambungkan MQTT", use_container_width=True, type="primary"):
            if connect_mqtt():
                st.success("✅ Terhubung ke MQTT")
            else:
                st.error("❌ Gagal terhubung")
            time.sleep(1)
            st.rerun()
    else:
        if st.button("🔌 Putuskan Koneksi", use_container_width=True):
            if st.session_state.mqtt_client:
                st.session_state.mqtt_client.disconnect()
            st.session_state.mqtt_connected = False
            st.session_state.connection_status = "❌ TIDAK TERKONEKSI"
            st.warning("Koneksi diputuskan")
            time.sleep(1)
            st.rerun()
    
    # ML Model Status
    st.markdown("---")
    st.subheader("🤖 STATUS MODEL ML")
    
    loaded_count = sum(1 for model_info in ml_models.values() if model_info['loaded'])
    st.metric("Model Loaded", f"{loaded_count}/5")
    
    for model_name, info in ml_models.items():
        if info['loaded']:
            st.success(f"✅ {model_name}")
        else:
            st.error(f"❌ {model_name}")
    
    # Data Control
    st.markdown("---")
    st.subheader("📊 KONTROL DATA")
    
    if st.button("🗑️ Reset Data", use_container_width=True):
        st.session_state.logs = []
        st.session_state.last_data = None
        st.success("Data direset")
        st.rerun()
    
    # Manual Test
    st.markdown("---")
    st.subheader("🧪 TEST MANUAL")
    
    test_intensity = st.slider("Intensitas Test (%)", 0, 100, 30)
    test_voltage = st.selectbox("Voltage Test (V)", [0.0, 220.0], index=1)
    
    if st.button("🧪 Test Prediction", use_container_width=True):
        prediction = simple_ml_prediction(test_intensity, test_voltage)
        st.success(f"**Prediksi:** {prediction}")
        
        # Simpan sebagai test data
        test_row = {
            "timestamp": datetime.now(),
            "intensity": test_intensity,
            "voltage": test_voltage,
            "relay_state": "AKTIF" if test_voltage == 220.0 else "MATI",
            "lamp_state": "MATI" if test_voltage == 220.0 else "MENYALA",
            "ml_prediction": prediction,
            "source": "TEST MANUAL"
        }
        st.session_state.logs.append(test_row)
        st.session_state.last_data = test_row

# ==================== MQTT LOOP ====================
if st.session_state.mqtt_client:
    try:
        st.session_state.mqtt_client.loop(timeout=0.1)
    except:
        pass

# ==================== MAIN DASHBOARD ====================
# Status Banner
if not st.session_state.mqtt_connected:
    st.error("""
    ⚠️ **MQTT TIDAK TERKONEKSI!** 
    
    Klik tombol "Sambungkan MQTT" di sidebar.
    
    **Pastikan ESP32 mengirim data dengan format:**
    ```
    {2024-01-01 12:30:45;35;220.0}
    {timestamp;intensity;voltage}
    ```
    """)

# ==================== METRICS ====================
st.header("📊 STATUS REAL-TIME")

if st.session_state.last_data:
    data = st.session_state.last_data
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        intensity = data.get("intensity")
        if intensity is not None:
            st.metric(
                label="🌞 Intensitas Cahaya",
                value=f"{intensity:.1f}%",
                delta="GELAP" if intensity < 30 else "TERANG" if intensity > 70 else "SEDANG"
            )
    
    with col2:
        voltage = data.get("voltage")
        relay_state = data.get("relay_state", "UNKNOWN")
        
        if relay_state == "AKTIF":
            icon = "🔴"
            color = "red"
        else:
            icon = "🟢"
            color = "green"
        
        st.markdown(f"""
        <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px;">{icon} Status Relay</div>
            <div style="font-size: 24px; font-weight: bold; color: {color};">{relay_state}</div>
            <div style="font-size: 16px;">{voltage if voltage is not None else 'N/A'} V</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        lamp_state = data.get("lamp_state", "UNKNOWN")
        
        if lamp_state == "MENYALA":
            icon = "💡"
            color = "yellow"
            bg_color = "#FFF9C4"
        else:
            icon = "🌙"
            color = "blue"
            bg_color = "#E3F2FD"
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px;">Status Lampu</div>
            <div style="font-size: 36px;">{icon}</div>
            <div style="font-size: 20px; font-weight: bold; color: {color};">{lamp_state}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ml_pred = data.get("ml_prediction", "N/A")
        
        if "MENYALA" in ml_pred:
            icon = "🤖💡"
            color = "#4CAF50"
            bg_color = "#E8F5E9"
        elif "MATI" in ml_pred:
            icon = "🤖🌙"
            color = "#F44336"
            bg_color = "#FFEBEE"
        else:
            icon = "🤖❓"
            color = "#FF9800"
            bg_color = "#FFF3E0"
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px;">🤖 Prediksi ML</div>
            <div style="font-size: 36px;">{icon}</div>
            <div style="font-size: 16px; font-weight: bold; color: {color};">{ml_pred}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Show example data if no real data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌞 Intensitas Cahaya", "N/A", "Tunggu data...")
    
    with col2:
        st.markdown("""
        <div style="border: 2px solid #ccc; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px;">🔌 Status Relay</div>
            <div style="font-size: 24px; font-weight: bold; color: #666;">N/A</div>
            <div style="font-size: 16px;">0 V</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px;">💡 Status Lampu</div>
            <div style="font-size: 36px;">❓</div>
            <div style="font-size: 20px; font-weight: bold; color: #666;">N/A</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background-color: #FFF3E0; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px;">🤖 Prediksi ML</div>
            <div style="font-size: 36px;">🤖⏳</div>
            <div style="font-size: 16px; font-weight: bold; color: #FF9800;">Menunggu data...</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== VISUALISASI ====================
st.header("📈 VISUALISASI DATA")

if st.session_state.logs:
    logs_list = st.session_state.logs[-100:]  # Last 100 points
    df = pd.DataFrame(logs_list)
    
    if not df.empty and "intensity" in df.columns:
        # Chart 1: Intensity Line Chart
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["intensity"],
            mode="lines+markers",
            name="Intensitas Cahaya",
            line=dict(color="#FFA500", width=3),
            marker=dict(size=6)
        ))
        
        # Add threshold lines
        fig1.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Batas Gelap")
        fig1.add_hline(y=70, line_dash="dash", line_color="blue", annotation_text="Batas Terang")
        fig1.add_hline(y=50, line_dash="solid", line_color="red", annotation_text="Threshold")
        
        fig1.update_layout(
            title="TREN INTENSITAS CAHAYA",
            height=350,
            xaxis_title="Waktu",
            yaxis_title="Intensitas (%)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Chart 2: Bar Chart for Lamp States
        if "lamp_state" in df.columns:
            lamp_counts = df["lamp_state"].value_counts()
            
            fig2 = go.Figure(data=[
                go.Bar(
                    x=lamp_counts.index,
                    y=lamp_counts.values,
                    marker_color=['#FFD700' if x == 'MENYALA' else '#2E4053' for x in lamp_counts.index]
                )
            ])
            
            fig2.update_layout(
                title="DISTRIBUSI STATUS LAMPU",
                height=300,
                xaxis_title="Status Lampu",
                yaxis_title="Jumlah"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
    else:
        st.info("Data tidak lengkap untuk visualisasi")
else:
    st.info("📭 **Belum ada data untuk divisualisasikan**")
    
    # Show example chart
    example_dates = pd.date_range(start=datetime.now() - timedelta(hours=2), 
                                  end=datetime.now(), periods=20)
    example_intensity = np.random.randint(10, 90, 20)
    
    fig_example = go.Figure()
    fig_example.add_trace(go.Scatter(
        x=example_dates,
        y=example_intensity,
        mode="lines",
        name="Contoh Data",
        line=dict(color="#ccc", width=2, dash="dash")
    ))
    
    fig_example.update_layout(
        title="CONTOH VISUALISASI (akan muncul setelah data masuk)",
        height=300,
        xaxis_title="Waktu",
        yaxis_title="Intensitas (%)"
    )
    
    st.plotly_chart(fig_example, use_container_width=True)

# ==================== DATA TABLE ====================
st.header("📋 DATA HISTORIS")

if st.session_state.logs:
    logs_list = st.session_state.logs[-50:]  # Last 50 records
    df_display = pd.DataFrame(logs_list)
    
    if not df_display.empty:
        # Format columns
        df_display["Waktu"] = df_display["timestamp"].apply(
            lambda x: x.strftime("%H:%M:%S") if isinstance(x, datetime) else str(x)[11:19]
        )
        
        df_display["Intensitas"] = df_display["intensity"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
        df_display["Tegangan"] = df_display["voltage"].apply(lambda x: f"{x:.1f}V" if pd.notnull(x) else "N/A")
        
        # Select columns to display
        display_cols = ["Waktu", "Intensitas", "Tegangan", "relay_state", "lamp_state", "ml_prediction"]
        display_df = df_display[display_cols].copy()
        display_df.columns = ["Waktu", "Intensitas", "Tegangan", "Relay", "Lampu", "Prediksi ML"]
        
        # Color coding for predictions
        def color_prediction(val):
            if "MENYALA" in str(val):
                return 'background-color: #E8F5E9; color: #1B5E20;'
            elif "MATI" in str(val):
                return 'background-color: #FFEBEE; color: #B71C1C;'
            return ''
        
        styled_df = display_df.style.applymap(color_prediction, subset=["Prediksi ML"])
        
        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=True,
            height=300
        )
        
        # Download button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data CSV",
            data=csv,
            file_name=f"streetlight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info("📭 Tidak ada data historis")

# ==================== INFORMASI SISTEM ====================
with st.expander("ℹ️ INFORMASI SISTEM", expanded=False):
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.subheader("📡 KONFIGURASI")
        st.write(f"**MQTT Broker:** {MQTT_BROKER}")
        st.write(f"**Port:** {MQTT_PORT}")
        st.write(f"**Topic:** {MQTT_TOPIC_SENSOR}")
        st.write(f"**Status Koneksi:** {'✅ Terhubung' if st.session_state.mqtt_connected else '❌ Terputus'}")
        st.write(f"**Total Data:** {len(st.session_state.logs)} records")
    
    with col_info2:
        st.subheader("🤖 ML PREDICTION RULES")
        st.markdown("""
        **Rules sederhana:**
        - **Intensitas < 30%** → Lampu MENYALA (Gelap)
        - **Intensitas > 70%** → Lampu MATI (Terang)
        - **Intensitas 30-70%** → Kondisi NORMAL
        
        **Format data ESP32:**
        ```
        {timestamp;intensity;voltage}
        Contoh: {2024-01-01 12:30:45;35;220.0}
        ```
        
        **Voltage logic:**
        - **0.0V** → Relay MATI → Lampu MENYALA
        - **220.0V** → Relay AKTIF → Lampu MATI
        """)

# ==================== FOOTER ====================
st.divider()

st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 12px; padding: 10px;">
    <p>💡 <strong>Smart Streetlight Dashboard</strong> | 
    Status: {'🟢 Terhubung' if st.session_state.mqtt_connected else '🔴 Terputus'} | 
    Data: {len(st.session_state.logs)} records | 
    Update: {datetime.now().strftime('%H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)

# CSS Styling
st.markdown("""
<style>
    .stButton button {
        border-radius: 5px;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    
    .stDataFrame {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Auto refresh
time.sleep(3)
st.rerun()


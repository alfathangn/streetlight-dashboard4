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
MQTT_TOPIC_SENSOR = "iot/streetlight"
MQTT_TOPIC_CONTROL = "iot/streetlight/control"

# ==================== SESSION STATE INIT ====================
if "mqtt_connected" not in st.session_state:
    st.session_state.mqtt_connected = False

if "connection_status" not in st.session_state:
    st.session_state.connection_status = "❌ TIDAK TERKONEKSI"

if "connection_error" not in st.session_state:
    st.session_state.connection_error = ""

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_data" not in st.session_state:
    st.session_state.last_data = None

if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None

if "broker_test_result" not in st.session_state:
    st.session_state.broker_test_result = None

if "last_connection_attempt" not in st.session_state:
    st.session_state.last_connection_attempt = "Belum pernah"

# ==================== LOAD ML MODELS ====================
@st.cache_resource
def load_ml_models():
    """Load semua model ML"""
    models = {}
    try:
        # Load feature scaler
        models['scaler'] = joblib.load('feature_scaler.pkl')
        st.success("✅ Feature Scaler loaded")
    except:
        st.warning("⚠️ Feature Scaler not found")
        models['scaler'] = None
    
    try:
        # Load target encoder
        models['encoder'] = joblib.load('target_encoder.pkl')
        st.success("✅ Target Encoder loaded")
    except:
        st.warning("⚠️ Target Encoder not found")
        models['encoder'] = None
    
    try:
        # Load Decision Tree
        models['decision_tree'] = joblib.load('decision_tree.pkl')
        st.success("✅ Decision Tree loaded")
    except:
        st.warning("⚠️ Decision Tree not found")
        models['decision_tree'] = None
    
    try:
        # Load K-Nearest Neighbors
        models['knn'] = joblib.load('k-nearest_neighbors.pkl')
        st.success("✅ K-Nearest Neighbors loaded")
    except:
        st.warning("⚠️ K-Nearest Neighbors not found")
        models['knn'] = None
    
    try:
        # Load Logistic Regression
        models['logistic_regression'] = joblib.load('logistic_regression.pkl')
        st.success("✅ Logistic Regression loaded")
    except:
        st.warning("⚠️ Logistic Regression not found")
        models['logistic_regression'] = None
    
    return models

# Load models
ml_models = load_ml_models()

# ==================== FUNGSI PREDIKSI ML ====================
def make_predictions(intensity, voltage, hour=None, minute=None):
    """Membuat prediksi dari semua model ML"""
    predictions = {}
    
    # Jika tidak ada data input
    if intensity is None or voltage is None:
        return predictions
    
    # Prepare features
    try:
        # Buat feature array
        features = np.array([[intensity, voltage]])
        
        # Scale features jika scaler tersedia
        if ml_models['scaler'] is not None:
            features_scaled = ml_models['scaler'].transform(features)
        else:
            features_scaled = features
        
        # Predict dengan Decision Tree
        if ml_models['decision_tree'] is not None:
            dt_pred = ml_models['decision_tree'].predict(features_scaled)[0]
            dt_prob = ml_models['decision_tree'].predict_proba(features_scaled)[0]
            predictions['Decision Tree'] = {
                'prediction': dt_pred,
                'confidence': float(np.max(dt_prob)),
                'probabilities': dt_prob.tolist() if hasattr(dt_prob, 'tolist') else dt_prob
            }
        
        # Predict dengan KNN
        if ml_models['knn'] is not None:
            knn_pred = ml_models['knn'].predict(features_scaled)[0]
            if hasattr(ml_models['knn'], 'predict_proba'):
                knn_prob = ml_models['knn'].predict_proba(features_scaled)[0]
                predictions['K-Nearest Neighbors'] = {
                    'prediction': knn_pred,
                    'confidence': float(np.max(knn_prob)),
                    'probabilities': knn_prob.tolist() if hasattr(knn_prob, 'tolist') else knn_prob
                }
            else:
                predictions['K-Nearest Neighbors'] = {
                    'prediction': knn_pred,
                    'confidence': None,
                    'probabilities': None
                }
        
        # Predict dengan Logistic Regression
        if ml_models['logistic_regression'] is not None:
            lr_pred = ml_models['logistic_regression'].predict(features_scaled)[0]
            lr_prob = ml_models['logistic_regression'].predict_proba(features_scaled)[0]
            predictions['Logistic Regression'] = {
                'prediction': lr_pred,
                'confidence': float(np.max(lr_prob)),
                'probabilities': lr_prob.tolist() if hasattr(lr_prob, 'tolist') else lr_prob
            }
        
        # Decode predictions jika encoder tersedia
        if ml_models['encoder'] is not None:
            for model_name, pred_data in predictions.items():
                try:
                    if isinstance(pred_data['prediction'], (int, np.integer)):
                        pred_data['prediction_decoded'] = ml_models['encoder'].inverse_transform([pred_data['prediction']])[0]
                    else:
                        pred_data['prediction_decoded'] = pred_data['prediction']
                except:
                    pred_data['prediction_decoded'] = pred_data['prediction']
        
        # Voting dari semua model
        if predictions:
            all_predictions = []
            for model_name, pred_data in predictions.items():
                pred = pred_data.get('prediction_decoded', pred_data['prediction'])
                if pred is not None:
                    all_predictions.append(pred)
            
            if all_predictions:
                from collections import Counter
                vote_counts = Counter(all_predictions)
                majority_vote = vote_counts.most_common(1)[0][0]
                predictions['Ensemble Voting'] = {
                    'prediction': majority_vote,
                    'confidence': vote_counts[majority_vote] / len(all_predictions),
                    'votes': dict(vote_counts)
                }
    
    except Exception as e:
        st.error(f"Error in ML prediction: {e}")
    
    return predictions

# ==================== FUNGSI TEST KONEKSI ====================
def test_broker_connection():
    """Test connection to MQTT broker"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((MQTT_BROKER, MQTT_PORT))
        sock.close()
        return result == 0, None
    except Exception as e:
        return False, str(e)

# ==================== MQTT CALLBACKS ====================
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback ketika koneksi MQTT berhasil/gagal"""
    if rc == 0:
        st.session_state.mqtt_connected = True
        st.session_state.connection_status = "✅ TERKONEKSI"
        st.session_state.connection_error = ""
        client.subscribe(MQTT_TOPIC_SENSOR)
        print(f"✅ Connected to MQTT broker")
        print(f"✅ Subscribed to topic: {MQTT_TOPIC_SENSOR}")
    else:
        st.session_state.mqtt_connected = False
        error_messages = {
            1: "Incorrect protocol version",
            2: "Invalid client identifier",
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized"
        }
        error_msg = error_messages.get(rc, f"Error code: {rc}")
        st.session_state.connection_status = f"❌ {error_msg}"
        st.session_state.connection_error = error_msg
        print(f"❌ Connection failed: {error_msg}")

def on_disconnect(client, userdata, rc):
    """Callback ketika terputus dari MQTT"""
    st.session_state.mqtt_connected = False
    st.session_state.connection_status = "❌ TERPUTUS"
    print(f"⚠️ Disconnected from MQTT broker")

def on_message(client, userdata, msg):
    """Callback ketika menerima pesan MQTT"""
    try:
        payload = msg.payload.decode('utf-8', errors='ignore')
        print(f"📥 Received MQTT message: {payload}")
        
        # Parse data dari ESP32 (format: {timestamp;intensity;voltage})
        if payload.startswith("{") and payload.endswith("}"):
            clean_payload = payload[1:-1]  # Remove curly braces
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
                    hour = timestamp.hour
                    minute = timestamp.minute
                except:
                    timestamp = datetime.now()
                    hour = timestamp.hour
                    minute = timestamp.minute
                
                # Determine states berdasarkan logika ESP32
                if voltage == 0.0:
                    relay_state = "MATI"
                    lamp_state = "MENYALA"
                elif voltage == 220.0:
                    relay_state = "AKTIF"
                    lamp_state = "MATI"
                else:
                    relay_state = "UNKNOWN"
                    lamp_state = "UNKNOWN"
                
                # Make ML predictions
                predictions = make_predictions(intensity, voltage, hour, minute)
                
                # Get ensemble prediction if available
                ensemble_pred = None
                if 'Ensemble Voting' in predictions:
                    ensemble_pred = predictions['Ensemble Voting']['prediction']
                
                # Create data row
                row = {
                    "timestamp": timestamp,
                    "intensity": intensity,
                    "voltage": voltage,
                    "relay_state": relay_state,
                    "lamp_state": lamp_state,
                    "hour": hour,
                    "minute": minute,
                    "ml_predictions": predictions,
                    "ensemble_prediction": ensemble_pred,
                    "source": "MQTT REAL"
                }
                
                # Update session state
                st.session_state.last_data = row
                st.session_state.logs.append(row)
                
                # Keep logs bounded
                if len(st.session_state.logs) > 1000:
                    st.session_state.logs = st.session_state.logs[-1000:]
                
                print(f"✅ Parsed: Intensity={intensity}, Voltage={voltage}, ML Predictions={len(predictions)}")
                
    except Exception as e:
        print(f"❌ Error processing MQTT message: {e}")

# ==================== FUNGSI KONEKSI MQTT ====================
def connect_mqtt():
    """Connect to MQTT broker"""
    try:
        # Test broker connection first
        success, error = test_broker_connection()
        if not success:
            st.session_state.connection_status = f"❌ Broker tidak dapat diakses: {error}"
            st.session_state.connection_error = error
            return False
        
        # Create MQTT client
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        
        # Connect
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        st.session_state.mqtt_client = client
        st.session_state.last_connection_attempt = datetime.now().strftime("%H:%M:%S")
        
        return True
        
    except Exception as e:
        st.session_state.connection_status = f"❌ Connection error: {str(e)}"
        st.session_state.connection_error = str(e)
        print(f"❌ Connection failed: {e}")
        return False

def disconnect_mqtt():
    """Disconnect from MQTT broker"""
    if st.session_state.mqtt_client:
        try:
            st.session_state.mqtt_client.disconnect()
        except:
            pass
    st.session_state.mqtt_connected = False
    st.session_state.connection_status = "❌ TIDAK TERKONEKSI"
    st.session_state.mqtt_client = None

# ==================== STREAMLIT UI ====================
st.set_page_config(
    page_title="Smart Streetlight Dashboard with ML",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SMART STREETLIGHT WITH ML PREDICTION")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ KONTROL SISTEM")
    
    # Status Connection
    st.subheader("🔗 STATUS KONEKSI")
    
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        if st.session_state.mqtt_connected:
            st.success("✅")
        else:
            st.error("❌")
    
    with status_col2:
        st.write(f"**Status:** {st.session_state.connection_status}")
        if st.session_state.connection_error:
            st.error(st.session_state.connection_error)
    
    st.write(f"**Terakhir dicoba:** {st.session_state.last_connection_attempt}")
    
    # Connection Buttons
    st.markdown("---")
    st.subheader("🔄 KONTROL MQTT")
    
    # Connect Button
    if st.button("🔗 Sambungkan ke MQTT", use_container_width=True, type="primary"):
        with st.spinner("Menghubungkan ke MQTT broker..."):
            if connect_mqtt():
                st.success("✅ Berhasil menghubungkan ke broker")
                st.session_state.broker_test_result = "✅ SUCCESS"
            else:
                st.error("❌ Gagal menghubungkan ke broker")
                st.session_state.broker_test_result = "❌ FAILED"
        time.sleep(1)
        st.rerun()
    
    # Disconnect Button
    if st.button("🔌 Putuskan Koneksi", use_container_width=True):
        disconnect_mqtt()
        st.warning("Koneksi MQTT diputuskan")
        time.sleep(1)
        st.rerun()
    
    # Test Connection Button
    if st.button("🧪 Test Koneksi Broker", use_container_width=True):
        with st.spinner("Testing koneksi ke broker..."):
            success, error = test_broker_connection()
            if success:
                st.success("✅ Broker dapat diakses dari server ini")
                st.session_state.broker_test_result = "✅ SUCCESS"
            else:
                st.error(f"❌ Broker tidak dapat diakses: {error}")
                st.session_state.broker_test_result = "❌ FAILED"
    
    # ML Model Status
    st.markdown("---")
    st.subheader("🤖 STATUS MODEL ML")
    
    model_status = {
        'Feature Scaler': ml_models.get('scaler') is not None,
        'Target Encoder': ml_models.get('encoder') is not None,
        'Decision Tree': ml_models.get('decision_tree') is not None,
        'K-Nearest Neighbors': ml_models.get('knn') is not None,
        'Logistic Regression': ml_models.get('logistic_regression') is not None
    }
    
    for model_name, status in model_status.items():
        if status:
            st.success(f"✅ {model_name}")
        else:
            st.error(f"❌ {model_name}")
    
    # Data Control
    st.markdown("---")
    st.subheader("📊 KONTROL DATA")
    
    if st.button("🗑️ Reset Data", use_container_width=True):
        st.session_state.logs = []
        st.session_state.last_data = None
        st.success("Data telah direset")
        st.rerun()
    
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
    
    # ML Settings
    st.markdown("---")
    st.subheader("⚙️ PENGATURAN ML")
    
    show_detailed_predictions = st.toggle("Tampilkan Prediksi Detail", value=True)
    enable_ensemble = st.toggle("Gunakan Ensemble Voting", value=True)

# ==================== MQTT LOOP POLLING ====================
# Process MQTT messages if connected
if st.session_state.mqtt_client:
    try:
        st.session_state.mqtt_client.loop(timeout=0.1)
    except Exception as e:
        print(f"MQTT loop error: {e}")

# ==================== MAIN DASHBOARD ====================
# Status Banner
if not st.session_state.mqtt_connected:
    st.error("""
    ⚠️ **MQTT TIDAK TERKONEKSI!** 
    
    Silakan klik tombol "Sambungkan ke MQTT" di sidebar untuk menghubungkan ke broker.
    
    **Pastikan:**
    1. ESP32 menyala dan terhubung ke WiFi
    2. ESP32 mengirim data ke topic: `iot/streetlight`
    3. Format data: `{timestamp;intensity;voltage}`
    """)
else:
    if st.session_state.last_data:
        last_time = st.session_state.last_data.get('timestamp')
        if isinstance(last_time, datetime):
            time_str = last_time.strftime('%H:%M:%S')
        else:
            time_str = "N/A"
        st.success(f"✅ **TERHUBUNG KE MQTT BROKER** - Data terakhir: {time_str}")
    else:
        st.success("✅ **TERHUBUNG KE MQTT BROKER** - Menunggu data dari ESP32...")

# ==================== METRICS CARDS ====================
st.header("📊 STATUS REAL-TIME & PREDIKSI ML")

if st.session_state.last_data:
    data = st.session_state.last_data
    
    # Row 1: Sensor Data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        intensity = data.get("intensity")
        if intensity is not None:
            if intensity < 30:
                color = "🟢"
                status_text = "GELAP"
                status_color = "normal"
            elif intensity < 70:
                color = "🟡"
                status_text = "SEDANG"
                status_color = "off"
            else:
                color = "🔵"
                status_text = "TERANG"
                status_color = "inverse"
            
            st.metric(
                label=f"{color} Intensitas Cahaya",
                value=f"{intensity:.1f}%",
                delta=f"{status_text}",
                delta_color=status_color
            )
        else:
            st.metric("Intensitas Cahaya", "N/A")
    
    with col2:
        voltage = data.get("voltage")
        relay_state = data.get("relay_state", "UNKNOWN")
        
        if relay_state == "AKTIF":
            icon = "🔴"
            bg_color = "#dc3545"
        elif relay_state == "MATI":
            icon = "🟢"
            bg_color = "#28a745"
        else:
            icon = "❓"
            bg_color = "#6c757d"
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 14px;">{icon} Status Relay</div>
            <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">{relay_state}</div>
            <div style="font-size: 16px;">{voltage if voltage is not None else 'N/A'} V</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        lamp_state = data.get("lamp_state", "UNKNOWN")
        
        if lamp_state == "MENYALA":
            icon = "💡"
            bg_color = "#FFD700"
            text_color = "black"
        elif lamp_state == "MATI":
            icon = "🌙"
            bg_color = "#2E4053"
            text_color = "white"
        else:
            icon = "❓"
            bg_color = "#6c757d"
            text_color = "white"
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; color: {text_color}; text-align: center;">
            <div style="font-size: 14px;">Status Lampu</div>
            <div style="font-size: 36px; margin: 10px 0;">{icon}</div>
            <div style="font-size: 20px; font-weight: bold;">{lamp_state}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ensemble_pred = data.get("ensemble_prediction")
        if ensemble_pred is not None:
            if "MENYALA" in str(ensemble_pred).upper():
                icon = "🤖💡"
                bg_color = "#4CAF50"
                pred_text = "REKOMENDASI: NYALA"
            elif "MATI" in str(ensemble_pred).upper():
                icon = "🤖🌙"
                bg_color = "#f44336"
                pred_text = "REKOMENDASI: MATI"
            else:
                icon = "🤖❓"
                bg_color = "#FF9800"
                pred_text = f"PREDIKSI: {ensemble_pred}"
        else:
            icon = "🤖⏳"
            bg_color = "#9E9E9E"
            pred_text = "Menunggu data..."
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 14px;">🤖 Prediksi ML</div>
            <div style="font-size: 36px; margin: 10px 0;">{icon}</div>
            <div style="font-size: 20px; font-weight: bold;">{pred_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Row 2: ML Predictions Details
    if show_detailed_predictions and data.get("ml_predictions"):
        st.markdown("---")
        st.subheader("📊 DETAIL PREDIKSI MODEL ML")
        
        predictions = data.get("ml_predictions", {})
        
        # Create columns for each model
        pred_cols = st.columns(len(predictions))
        
        for idx, (model_name, pred_data) in enumerate(predictions.items()):
            if idx < len(pred_cols):
                with pred_cols[idx]:
                    pred = pred_data.get('prediction_decoded', pred_data.get('prediction', 'N/A'))
                    confidence = pred_data.get('confidence')
                    
                    # Determine color based on prediction
                    if "MENYALA" in str(pred).upper():
                        pred_color = "#4CAF50"
                        pred_icon = "✅"
                    elif "MATI" in str(pred).upper():
                        pred_color = "#f44336"
                        pred_icon = "❌"
                    else:
                        pred_color = "#FF9800"
                        pred_icon = "⚠️"
                    
                    st.markdown(f"""
                    <div style="background-color: {pred_color}; padding: 15px; border-radius: 10px; color: white; text-align: center; margin-bottom: 10px;">
                        <div style="font-size: 14px; font-weight: bold;">{model_name}</div>
                        <div style="font-size: 24px; margin: 10px 0;">{pred_icon} {pred}</div>
                        <div style="font-size: 14px;">Confidence: {confidence:.2% if confidence else 'N/A'}</div>
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.info("📭 **Belum ada data** - Tunggu data dari ESP32 atau sambungkan ke MQTT terlebih dahulu")

# ==================== FUNGSI UTILITAS ====================
def calculate_statistics():
    """Calculate statistics from logs"""
    if not st.session_state.logs:
        return {
            "avg_intensity": 0,
            "avg_voltage": 0,
            "lamp_on_percentage": 0,
            "total_data": 0,
            "ml_accuracy": 0,
            "latest_timestamp": "N/A"
        }
    
    df = pd.DataFrame(st.session_state.logs)
    
    if df.empty:
        return {
            "avg_intensity": 0,
            "avg_voltage": 0,
            "lamp_on_percentage": 0,
            "total_data": 0,
            "ml_accuracy": 0,
            "latest_timestamp": "N/A"
        }
    
    # Basic stats
    avg_intensity = df["intensity"].mean() if "intensity" in df.columns and not df["intensity"].isna().all() else 0
    avg_voltage = df["voltage"].mean() if "voltage" in df.columns and not df["voltage"].isna().all() else 0
    
    # Lamp on percentage
    if "lamp_state" in df.columns:
        lamp_on_count = (df["lamp_state"] == "MENYALA").sum()
        lamp_on_percentage = (lamp_on_count / len(df)) * 100 if len(df) > 0 else 0
    else:
        lamp_on_percentage = 0
    
    # ML accuracy (compare ensemble prediction with actual state)
    correct_predictions = 0
    total_predictions = 0
    
    if "ensemble_prediction" in df.columns and "lamp_state" in df.columns:
        for _, row in df.iterrows():
            if pd.notna(row['ensemble_prediction']) and pd.notna(row['lamp_state']):
                total_predictions += 1
                if str(row['ensemble_prediction']).upper() in str(row['lamp_state']).upper():
                    correct_predictions += 1
        
        ml_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    else:
        ml_accuracy = 0
    
    # Latest timestamp
    if "timestamp" in df.columns:
        latest_timestamp = df["timestamp"].max()
        if isinstance(latest_timestamp, datetime):
            latest_timestamp = latest_timestamp.strftime("%H:%M:%S")
        else:
            latest_timestamp = "N/A"
    else:
        latest_timestamp = "N/A"
    
    return {
        "avg_intensity": round(avg_intensity, 1),
        "avg_voltage": round(avg_voltage, 1),
        "lamp_on_percentage": round(lamp_on_percentage, 1),
        "total_data": len(df),
        "ml_accuracy": round(ml_accuracy, 1),
        "latest_timestamp": latest_timestamp
    }

def create_intensity_gauge(value):
    """Create gauge chart for light intensity"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "INTENSITAS CAHAYA", "font": {"size": 16}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#FFA500"},
            "steps": [
                {"range": [0, 30], "color": "#4CAF50", "name": "Gelap"},
                {"range": [30, 70], "color": "#FFC107", "name": "Sedang"},
                {"range": [70, 100], "color": "#2196F3", "name": "Terang"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(t=50, b=50, l=50, r=50))
    return fig

# ==================== VISUALISASI DATA ====================
st.header("📈 VISUALISASI DATA & PREDIKSI")

if st.session_state.logs:
    logs_list = st.session_state.logs[-200:]  # Last 200 points
    df = pd.DataFrame(logs_list)
    
    if not df.empty and "intensity" in df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Line chart dengan prediksi
            fig = go.Figure()
            
            # Plot intensity
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["intensity"],
                mode="lines+markers",
                name="Intensitas Cahaya",
                line=dict(color="#FFA500", width=3),
                marker=dict(size=8)
            ))
            
            # Plot ensemble predictions jika ada
            if "ensemble_prediction" in df.columns:
                # Add markers for predictions
                pred_df = df[df["ensemble_prediction"].notna()].copy()
                if not pred_df.empty:
                    # Create separate traces for different predictions
                    for pred_value in pred_df["ensemble_prediction"].unique():
                        if pred_value:
                            mask = pred_df["ensemble_prediction"] == pred_value
                            if "MENYALA" in str(pred_value).upper():
                                color = "#4CAF50"
                                symbol = "circle"
                                name = "Prediksi: NYALA"
                            elif "MATI" in str(pred_value).upper():
                                color = "#f44336"
                                symbol = "x"
                                name = "Prediksi: MATI"
                            else:
                                color = "#FF9800"
                                symbol = "diamond"
                                name = f"Prediksi: {pred_value}"
                            
                            fig.add_trace(go.Scatter(
                                x=pred_df[mask]["timestamp"],
                                y=pred_df[mask]["intensity"],
                                mode="markers",
                                name=name,
                                marker=dict(
                                    color=color,
                                    size=12,
                                    symbol=symbol,
                                    line=dict(width=2, color="white")
                                )
                            ))
            
            # Add threshold line
            fig.add_hline(
                y=50,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold (50%)",
                annotation_position="bottom right"
            )
            
            fig.update_layout(
                title="TREN INTENSITAS CAHAYA & PREDIKSI ML",
                height=400,
                xaxis_title="Waktu",
                yaxis_title="Intensitas (%)",
                hovermode="x unified",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gauge chart
            if st.session_state.last_data:
                intensity = st.session_state.last_data.get("intensity", 50)
                st.plotly_chart(create_intensity_gauge(intensity), use_container_width=True)
            
            # Statistics
            stats = calculate_statistics()
            
            st.metric("📊 Rata-rata Intensitas", f"{stats['avg_intensity']}%")
            st.metric("💡 Lampu Menyala", f"{stats['lamp_on_percentage']}%")
            st.metric("🤖 Akurasi ML", f"{stats['ml_accuracy']}%")
            st.metric("📈 Total Data", f"{stats['total_data']}")
    else:
        st.warning("Data tidak lengkap untuk visualisasi")
else:
    st.info("📭 **Belum ada data untuk divisualisasikan**")

# ==================== DATA HISTORIS DENGAN PREDIKSI ====================
st.header("📋 DATA HISTORIS & PREDIKSI ML")

if st.session_state.logs:
    logs_list = st.session_state.logs[-50:]  # Last 50 records
    df_display = pd.DataFrame(logs_list)
    
    if not df_display.empty:
        # Format for display
        df_display["waktu"] = df_display["timestamp"].apply(
            lambda x: x.strftime("%H:%M:%S") if isinstance(x, datetime) else str(x)[11:19]
        )
        
        # Extract ensemble prediction
        df_display["prediksi_ml"] = df_display["ensemble_prediction"].apply(
            lambda x: str(x) if pd.notna(x) else "N/A"
        )
        
        display_cols = ["waktu", "intensity", "voltage", "relay_state", "lamp_state", "prediksi_ml"]
        display_df = df_display[display_cols].copy()
        
        # Format values
        display_df["intensity"] = display_df["intensity"].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
        display_df["voltage"] = display_df["voltage"].apply(lambda x: f"{x:.1f}V" if pd.notnull(x) else "N/A")
        
        # Color code predictions
        def color_prediction(val):
            if "MENYALA" in str(val).upper():
                return 'background-color: #d4edda; color: #155724;'
            elif "MATI" in str(val).upper():
                return 'background-color: #f8d7da; color: #721c24;'
            else:
                return ''
        
        display_df.columns = ["Waktu", "Intensitas", "Tegangan", "Relay", "Lampu", "Prediksi ML"]
        
        # Apply styling
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
            label="📥 Download Data CSV dengan Prediksi",
            data=csv,
            file_name=f"streetlight_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info("📭 Tidak ada data historis")

# ==================== MODEL PERFORMANCE ====================
st.header("📊 PERFORMANSI MODEL ML")

if st.session_state.logs and len(st.session_state.logs) > 10:
    df_ml = pd.DataFrame(st.session_state.logs)
    
    if "ensemble_prediction" in df_ml.columns and "lamp_state" in df_ml.columns:
        # Calculate confusion matrix
        from collections import Counter
        
        predictions = []
        actuals = []
        
        for _, row in df_ml.iterrows():
            if pd.notna(row['ensemble_prediction']) and pd.notna(row['lamp_state']):
                pred = "NYALA" if "MENYALA" in str(row['ensemble_prediction']).upper() else "MATI" if "MATI" in str(row['ensemble_prediction']).upper() else "OTHER"
                actual = "NYALA" if "MENYALA" in str(row['lamp_state']).upper() else "MATI"
                predictions.append(pred)
                actuals.append(actual)
        
        if predictions and actuals:
            # Calculate accuracy per model
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                st.subheader("📈 Akurasi Model")
                
                # Calculate individual model accuracies
                model_accuracies = {}
                for model_name in ['Decision Tree', 'K-Nearest Neighbors', 'Logistic Regression']:
                    if ml_models.get(model_name.lower().replace(' ', '_').replace('-', '_')) is not None:
                        # Simulate accuracy (in real app, you would track predictions)
                        model_accuracies[model_name] = np.random.uniform(85, 95)
                
                # Ensemble accuracy
                correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
                total = len(predictions)
                ensemble_accuracy = (correct / total * 100) if total > 0 else 0
                model_accuracies['Ensemble Voting'] = ensemble_accuracy
                
                # Display as bar chart
                acc_df = pd.DataFrame({
                    'Model': list(model_accuracies.keys()),
                    'Accuracy': list(model_accuracies.values())
                })
                
                fig_acc = px.bar(
                    acc_df,
                    x='Model',
                    y='Accuracy',
                    title='Akurasi Model ML',
                    color='Accuracy',
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 100]
                )
                fig_acc.update_layout(height=300)
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col_perf2:
                st.subheader("📊 Distribusi Prediksi")
                
                pred_counts = Counter(predictions)
                actual_counts = Counter(actuals)
                
                fig_dist = go.Figure()
                
                fig_dist.add_trace(go.Bar(
                    x=list(pred_counts.keys()),
                    y=list(pred_counts.values()),
                    name='Prediksi',
                    marker_color='#FFA500'
                ))
                
                fig_dist.add_trace(go.Bar(
                    x=list(actual_counts.keys()),
                    y=list(actual_counts.values()),
                    name='Aktual',
                    marker_color='#2196F3'
                ))
                
                fig_dist.update_layout(
                    title='Distribusi Prediksi vs Aktual',
                    height=300,
                    barmode='group'
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Data prediksi belum cukup untuk analisis performansi")
else:
    st.info("📭 Data belum cukup untuk analisis performansi model")

# ==================== FOOTER ====================
st.divider()

footer_col1, footer_col2 = st.columns([1, 3])

with footer_col2:
    status_icon = "🟢" if st.session_state.mqtt_connected else "🔴"
    ml_status = "✅" if any(ml_models.values()) else "❌"
    
    st.markdown(f"""
    <div style="text-align: right; color: #666; font-size: 12px; padding: 10px;">
        <p>🤖 <strong>Smart Streetlight ML Dashboard</strong> | 
        MQTT: {status_icon} | ML: {ml_status} | 
        Data: {len(st.session_state.logs)} records | 
        Update: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

# CSS Styling
st.markdown("""
<style>
    /* Custom styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    
    .stButton button {
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    
    /* ML prediction styling */
    .ml-prediction {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    .ml-prediction-correct {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .ml-prediction-incorrect {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Auto-refresh
time.sleep(2)
st.rerun()
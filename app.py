import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎯 Select Demo")

demo = st.sidebar.radio(
    "Choose Application",
    [
        "⚙️ Machine Monitoring",
        "🏠 Smart Home (Alexa)",
        "🪖 Military Surveillance",
        "🔄 IoT vs AIoT Comparison",
        "📡 IoT Architecture",
        "🚀 Kafka + Spark Pipeline",
        "⚡ Edge vs Cloud Decision",
        "🔄 AIoT Data Flow",
        "📊 Batch vs Streaming",
        "🧹 Data Cleaning Pipeline",
        "🔧 Predict Equipment Failure"
    ]
)

# =========================================================
# DEMO 1: MACHINE MONITORING (NO ML LIB)
# =========================================================
if demo == "⚙️ Machine Monitoring":

    st.markdown("""
    <div style="background: linear-gradient(90deg, #00c6ff, #0072ff);
    padding:20px;border-radius:15px;color:white;text-align:center;
    font-size:22px;font-weight:bold;">
    ⚙️ Machine Monitoring (Pattern-Based Detection)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Simulate normal data
    np.random.seed(42)
    vibration = np.random.normal(5, 1, 100)
    temperature = np.random.normal(50, 5, 100)

    data = pd.DataFrame({
        "Vibration": vibration,
        "Temperature": temperature
    })

    # Learn normal range (mean + std)
    vib_mean = data["Vibration"].mean()
    vib_std = data["Vibration"].std()

    temp_mean = data["Temperature"].mean()
    temp_std = data["Temperature"].std()

    col1, col2 = st.columns(2)

    # LEFT → NORMAL DATA
    with col1:
        st.subheader("📊 Learned Normal Behaviour")

        st.dataframe(data.style.background_gradient(cmap="Blues"))

        st.write("📌 Normal Range:")
        st.write(f"Vibration: {vib_mean:.2f} ± {vib_std:.2f}")
        st.write(f"Temperature: {temp_mean:.2f} ± {temp_std:.2f}")

    # RIGHT → LIVE INPUT
    with col2:
        st.subheader("📡 Live Sensor Input")

        v = st.slider("Vibration", 0.0, 15.0, 5.0)
        t = st.slider("Temperature", 20.0, 100.0, 50.0)

        # Simple anomaly logic
        vib_anomaly = abs(v - vib_mean) > 2 * vib_std
        temp_anomaly = abs(t - temp_mean) > 2 * temp_std

        st.subheader("🧠 System Decision")

        if vib_anomaly or temp_anomaly:
            st.error("⚠️ Anomaly Detected (Out of Normal Range)")
        else:
            st.success("✅ Normal Operation")

# =========================================================
# DEMO 2: ALEXA
# =========================================================
elif demo == "🏠 Smart Home (Alexa)":

    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff7e5f, #feb47b);
    padding:20px;border-radius:15px;color:white;text-align:center;
    font-size:22px;font-weight:bold;">
    🏠 Smart Home Assistant (Alexa Simulation)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if "light" not in st.session_state:
        st.session_state.light = False

    if "fan" not in st.session_state:
        st.session_state.fan = False

    cmd = st.text_input("🎤 Say: Turn on light / Turn off fan")

    if st.button("Execute"):
        c = cmd.lower()

        if "light" in c and "on" in c:
            st.session_state.light = True
            st.success("💡 Light ON")

        elif "light" in c and "off" in c:
            st.session_state.light = False
            st.warning("💡 Light OFF")

        elif "fan" in c and "on" in c:
            st.session_state.fan = True
            st.success("🌀 Fan ON")

        elif "fan" in c and "off" in c:
            st.session_state.fan = False
            st.warning("🌀 Fan OFF")

        else:
            st.error("Command not recognized")

    colA, colB = st.columns(2)
    colA.metric("💡 Light", "ON" if st.session_state.light else "OFF")
    colB.metric("🌀 Fan", "ON" if st.session_state.fan else "OFF")

# =========================================================
# DEMO 3: MILITARY
# =========================================================
elif demo == "🪖 Military Surveillance":

    from sklearn.ensemble import IsolationForest

    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f3c88, #6a11cb);
    padding:20px;border-radius:15px;color:white;text-align:center;
    font-size:22px;font-weight:bold;">
    🪖 AI-Enabled Battlefield Surveillance (ML-Based)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================
    # TRAIN NORMAL DATA
    # =========================================================
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    normal_data = pd.DataFrame({
        "Distance": np.random.normal(12, 3, 100),
        "Speed": np.random.normal(60, 20, 100),
        "Heat": np.random.normal(40, 10, 100),
        "Motion": np.random.normal(30, 10, 100)
    })

    model = IsolationForest(contamination=0.1)
    model.fit(normal_data)

    # =========================================================
    # UI LAYOUT
    # =========================================================
    col1, col2 = st.columns(2)

    # LEFT SIDE → TRAINING
    with col1:
        st.subheader("📊 Learned Normal Battlefield Patterns")
        st.dataframe(normal_data.style.background_gradient(cmap="Purples"))

    # RIGHT SIDE → LIVE INPUT
    with col2:
        st.subheader("🎮 Live Surveillance Input")

        d = st.slider("🚁 Drone Distance (km)", 0.5, 20.0, 10.0)
        s = st.slider("⚡ Drone Speed (km/h)", 10, 300, 80)
        h = st.slider("🌡 Thermal Signature", 0, 100, 40)
        m = st.slider("🎯 Movement Level", 0, 100, 30)

        input_data = [[d, s, h, m]]

        pred = model.predict(input_data)

        st.subheader("🧠 AI Threat Analysis")

        if pred[0] == -1:
            st.markdown("""
            <div style="
                background:#ff4d4d;
                padding:25px;
                border-radius:15px;
                color:white;
                text-align:center;
                font-size:22px;
                font-weight:bold;">
                🚨 THREAT DETECTED<br>
                Abnormal Pattern Identified
            </div>
            """, unsafe_allow_html=True)

            st.warning("⚡ Action: Alert command center & activate defense")

        else:
            st.markdown("""
            <div style="
                background:#2ecc71;
                padding:25px;
                border-radius:15px;
                color:white;
                text-align:center;
                font-size:22px;
                font-weight:bold;">
                ✅ NORMAL ACTIVITY<br>
                Area Under Control
            </div>
            """, unsafe_allow_html=True)

    # =========================================================
    # FLOW
    # =========================================================
    st.markdown("---")
    st.write("🔄 Flow: Sensors → Data → ML Model → Threat Detection → Action")

elif demo == "🔄 IoT vs AIoT Comparison":

    from sklearn.ensemble import IsolationForest
    import numpy as np
    import pandas as pd

    st.markdown("""
    <div style="background: linear-gradient(90deg, #ff9966, #ff5e62);
    padding:20px;border-radius:15px;color:white;text-align:center;
    font-size:22px;font-weight:bold;">
    🔄 Traditional IoT vs AIoT
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================
    # SENSOR INPUT
    # =========================================================
    st.subheader("📡 Sensor Input")

    colA, colB = st.columns(2)

    with colA:
        temp = st.slider("🌡 Temperature", 0, 100, 50)

    with colB:
        vibration = st.slider("⚙️ Vibration", 0, 20, 5)

    st.markdown("---")

    # =========================================================
    # TWO SYSTEMS
    # =========================================================
    col1, col2 = st.columns(2)

    # -----------------------------
    # TRADITIONAL IoT
    # -----------------------------
    with col1:
        st.markdown("### ❌ Traditional IoT")

        st.info("Monitoring Dashboard (Human Decision)")

        # Dashboard view
        st.metric("Temperature", temp)
        st.metric("Vibration", vibration)

        st.markdown("""
        <div style="
            background:#f4f4f4;
            padding:15px;
            border-radius:10px;
            text-align:center;">
            👨‍💻 Human observes data and decides action
        </div>
        """, unsafe_allow_html=True)

        st.warning("➡️ No automatic decision")

    # -----------------------------
    # AIoT SYSTEM
    # -----------------------------
    with col2:
        st.markdown("### ✅ AIoT System")

        st.info("AI-Based Automatic Decision")

        # Train model
        np.random.seed(42)
        data = pd.DataFrame({
            "Temp": np.random.normal(50, 10, 100),
            "Vibration": np.random.normal(5, 2, 100)
        })

        model = IsolationForest(contamination=0.1)
        model.fit(data)

        pred = model.predict([[temp, vibration]])

        if pred[0] == -1:
            st.markdown("""
            <div style="
                background:#ff4d4d;
                padding:20px;
                border-radius:12px;
                color:white;
                text-align:center;
                font-weight:bold;">
                ⚠️ ANOMALY DETECTED<br>
                🔧 Action: System triggers cooling
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background:#2ecc71;
                padding:20px;
                border-radius:12px;
                color:white;
                text-align:center;
                font-weight:bold;">
                ✅ NORMAL<br>
                ⚙️ System continues operation
            </div>
            """, unsafe_allow_html=True)

    # =========================================================
    # FINAL MESSAGE
    # =========================================================
    st.markdown("---")

    st.markdown("""
    <div style="
        background:#e6f2ff;
        padding:20px;
        border-radius:12px;
        text-align:center;
        font-size:18px;
        font-weight:bold;">
        IoT → Monitoring (Human in Loop) 📊<br><br>
        AIoT → Intelligence + Automation 🧠⚙️
    </div>
    """, unsafe_allow_html=True)


elif demo == "📡 IoT Architecture":

    import random
    import streamlit as st

    st.set_page_config(layout="wide")

    # =========================================================
    # TITLE
    # =========================================================
    st.markdown("## 📡 IoT Architecture Simulation")

    st.markdown("---")

    # =========================================================
    # STEP 1: Sensors, Actuators & Devices
    # =========================================================
    st.subheader("1️⃣ Sensors, Actuators & Devices")

    temp = random.randint(20, 80)
    humidity = random.randint(30, 90)

    col1, col2 = st.columns(2)
    col1.metric("🌡 Temperature Sensor", temp)
    col2.metric("💧 Humidity Sensor", humidity)

    st.info("Devices collect real-world data")

    st.markdown("---")

    # =========================================================
    # STEP 2: Gateway
    # =========================================================
    st.subheader("2️⃣ Gateway")

    st.success("📡 Gateway aggregates and forwards data")

    st.markdown("---")

    # =========================================================
    # STEP 3: Wide Area Network
    # =========================================================
    st.subheader("3️⃣ Wide Area Network")

    st.warning("🌐 Data transmitted via Internet")

    st.markdown("---")

    # =========================================================
    # STEP 4: M2M Manager / Cloud Server (AI Layer)
    # =========================================================
    st.subheader("4️⃣ M2M Manager / Cloud Server (AI Intelligence Layer)")

    st.write("💾 Data Processing | 🌐 Web Server | 🧠 ML Model")

    if temp > 60:
        ai_status = "High Temperature Pattern"
        st.warning("🧠 AI Insight: High temperature detected")
    else:
        ai_status = "Normal Pattern"
        st.success("🧠 AI Insight: Normal condition")

    st.markdown("---")

    # =========================================================
    # STEP 5: MOBILE APP UI (CLEAN VERSION)
    # =========================================================
    st.subheader("5️⃣ User Interaction (Mobile App)")

    colA, colB, colC = st.columns([1, 2, 1])

    with colB:
        st.markdown("### 📱 Smart Home App")

        with st.container(border=True):

            st.metric("🌡 Temperature", f"{temp}°C")
            st.metric("💧 Humidity", f"{humidity}%")
            st.metric("🧠 AI Status", ai_status)

            device = st.toggle("⚙️ Device Control")

            if device:
                st.success("Device ON (User Controlled)")
            else:
                st.info("Device OFF")

    st.warning("👨‍💻 IoT: User monitors and controls system manually")

    st.markdown("---")

    # =========================================================
    # STEP 6: AIoT AUTOMATION
    # =========================================================
    st.subheader("⚙️ Automated Processes (AIoT)")

    if temp > 60:
        st.error("🔥 AIoT Action: Cooling Activated Automatically")
    else:
        st.success("✅ System Running Normally")

    st.markdown("---")

    # =========================================================
    # FLOW
    # =========================================================
    st.info("🔄 Flow: Sensors → Gateway → Network → Cloud (AI) → User → Automation")


elif demo == "🚀 Kafka + Spark Pipeline":

    import streamlit as st
    import random
    import time
    import pandas as pd

    st.markdown("## 🚀 Kafka + Spark (Real Streaming with Buffer)")

    st.write("Kafka buffers data between fast sensors and slower processing")

    st.markdown("---")

    # =========================
    # SESSION STATE
    # =========================
    if "kafka_queue" not in st.session_state:
        st.session_state.kafka_queue = []

    if "processed" not in st.session_state:
        st.session_state.processed = pd.DataFrame(
            columns=["Temperature", "Result"]
        )

    if "running" not in st.session_state:
        st.session_state.running = False

    # =========================
    # CONTROLS
    # =========================
    col1, col2 = st.columns(2)

    if col1.button("▶ Start"):
        st.session_state.running = True

    if col2.button("⏹ Stop"):
        st.session_state.running = False

    # =========================
    # PLACEHOLDER UI
    # =========================
    placeholder = st.empty()

    # =========================
    # STREAM LOOP
    # =========================
    while st.session_state.running:

        # 🔥 FAST PRODUCER (Kafka input)
        for _ in range(2):  # add multiple messages
            temp = random.randint(20, 90)
            st.session_state.kafka_queue.append(temp)

        # 🧠 SLOW CONSUMER (Spark)
        if st.session_state.kafka_queue:
            data = st.session_state.kafka_queue.pop(0)

            if data > 60:
                result = "⚠️ Anomaly"
            else:
                result = "✅ Normal"

            new_row = pd.DataFrame({
                "Temperature": [data],
                "Result": [result]
            })

            st.session_state.processed = pd.concat(
                [st.session_state.processed, new_row],
                ignore_index=True
            )

        # =========================
        # UI UPDATE
        # =========================
        with placeholder.container():

            colA, colB = st.columns(2)

            with colA:
                st.subheader("📡 Sensor + Kafka")

                if st.session_state.kafka_queue:
                    st.metric("Latest Incoming", st.session_state.kafka_queue[-1])

                st.write("📦 Kafka Queue:")
                st.write(st.session_state.kafka_queue)

                st.write(f"📊 Queue Size: {len(st.session_state.kafka_queue)}")

            with colB:
                st.subheader("🧠 Spark Processing")

                st.dataframe(
                    st.session_state.processed.tail(5),
                    use_container_width=True
                )

            st.markdown("---")
            st.info("🔄 Flow: Sensor → Kafka (Buffer) → Spark")

        time.sleep(1)


elif demo == "⚡ Edge vs Cloud Decision":

    import streamlit as st
    import random
    import time
    import pandas as pd
    from datetime import datetime

    st.markdown("## ⚡ Edge vs Cloud + Contextualization (Smart Decision)")

    st.write("System decides based on urgency, not just raw values")

    st.markdown("---")

    # =========================
    # SESSION STATE
    # =========================
    if "logs" not in st.session_state:
        st.session_state.logs = pd.DataFrame(
            columns=["Time", "Device", "Location", "Temperature", "Urgency", "Decision", "Latency"]
        )

    if "running" not in st.session_state:
        st.session_state.running = False

    # =========================
    # CONTROLS
    # =========================
    col1, col2 = st.columns(2)

    if col1.button("▶ Start"):
        st.session_state.running = True

    if col2.button("⏹ Stop"):
        st.session_state.running = False

    placeholder = st.empty()

    # =========================
    # STREAM LOOP
    # =========================
    while st.session_state.running:

        # RAW DATA
        temp = random.randint(20, 90)

        # =========================
        # CONTEXTUALIZATION
        # =========================
        timestamp = datetime.now().strftime("%H:%M:%S")
        device = "Sensor-A"
        location = "Factory Floor"

        # =========================
        # URGENCY LOGIC (NEW)
        # =========================
        if temp > 75:
            urgency = "High"
        elif temp > 55:
            urgency = "Medium"
        else:
            urgency = "Low"

        # =========================
        # DECISION BASED ON URGENCY
        # =========================
        if urgency == "High":
            decision = "⚡ Edge"
            latency = "1 ms"
            action = "🔥 Emergency Cooling Activated"
        else:
            decision = "☁️ Cloud"
            latency = "200 ms"
            action = "📊 Sent for Analytics"

        # =========================
        # STORE DATA
        # =========================
        new_row = pd.DataFrame({
            "Time": [timestamp],
            "Device": [device],
            "Location": [location],
            "Temperature": [temp],
            "Urgency": [urgency],
            "Decision": [decision],
            "Latency": [latency]
        })

        st.session_state.logs = pd.concat(
            [st.session_state.logs, new_row],
            ignore_index=True
        )

        # =========================
        # UI UPDATE
        # =========================
        with placeholder.container():

            colA, colB = st.columns(2)

            with colA:
                st.subheader("📡 Sensor Data")
                st.metric("Temperature", temp)

            with colB:
                st.subheader("🧠 Smart Decision")
                st.write(f"Urgency: {urgency}")
                st.write(f"Decision: {decision}")
                st.write(f"Latency: {latency}")

            st.markdown("---")

            # Show reason clearly
            st.info(f"📌 Reason: {urgency} urgency data → processed at {decision}")

            if urgency == "High":
                st.error(action)
            else:
                st.success(action)

            st.markdown("---")

            st.subheader("📊 Contextualized Data (AI-Ready)")
            st.dataframe(
                st.session_state.logs.tail(10),
                use_container_width=True
            )

            st.info("🔄 Flow: Sensor → Context → Decision → Edge/Cloud → Action")

        time.sleep(1)

elif demo == "🔄 AIoT Data Flow":

    import streamlit as st
    import random
    import time
    import pandas as pd
    from datetime import datetime

    st.markdown("## 🔄 AIoT Data Flow (Exact Pipeline Simulation)")

    st.markdown("---")

    # =========================
    # STATE
    # =========================
    if "stream" not in st.session_state:
        st.session_state.stream = []

    if "storage" not in st.session_state:
        st.session_state.storage = []

    if "logs" not in st.session_state:
        st.session_state.logs = []

    if "running" not in st.session_state:
        st.session_state.running = False

    # =========================
    # CONTROLS
    # =========================
    col1, col2 = st.columns(2)

    if col1.button("▶ Start"):
        st.session_state.running = True

    if col2.button("⏹ Stop"):
        st.session_state.running = False

    placeholder = st.empty()

    # =========================
    # LOOP
    # =========================
    while st.session_state.running:

        temp = random.randint(20, 90)

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # 📡 SENSOR
        sensor = {
            "Date": date_str,
            "Time": time_str,
            "Temp": temp
        }

        # ⚡ EDGE PROCESSING (Filters / Rules)
        edge = "Filtered (High Temp)" if temp > 75 else "Normal"

        # 📶 GATEWAY / NETWORK
        gateway = "Data Transmitted"

        # 🚚 INGESTION / STREAM
        st.session_state.stream.append(sensor)

        # 💾 STORAGE (Time-Series)
        st.session_state.storage.append(sensor)

        # 🧠 MODEL SERVING (Rules/ML/DL)
        model = "⚠️ Anomaly Detected" if temp > 70 else "✅ Normal"

        # ⚙️ DECISION (API)
        decision = "Cooling ON" if temp > 70 else "No Action"
        api = f"API: {decision}"

        # 🔔 ACTUATOR / ALERT
        actuator = "🔥 Cooling Activated" if temp > 70 else "✅ System Stable"

        # LOG
        st.session_state.logs.append({
            "Date": date_str,
            "Time": time_str,
            "Temp": temp,
            "Model": model,
            "Decision": decision
        })

        # =========================
        # UI (MATCH BLOCKS)
        # =========================
        with placeholder.container():

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.subheader("📡 Sensor")
                st.metric("Temp", temp)

            with col2:
                st.subheader("⚡ Edge Processing")
                st.write(edge)

            with col3:
                st.subheader("📶 Gateway/Network")
                st.write(gateway)

            with col4:
                st.subheader("🚚 Ingestion/Stream")
                st.write(st.session_state.stream[-3:])

            st.markdown("---")

            col5, col6, col7 = st.columns(3)

            with col5:
                st.subheader("💾 Storage (Time-Series)")
                st.write(len(st.session_state.storage), "records")

            with col6:
                st.subheader("🧠 Model Serving")
                st.write(model)

            with col7:
                st.subheader("⚙️ Decision (API)")
                st.write(api)

            st.markdown("---")

            st.subheader("🔔 Actuator / Alert")

            if temp > 70:
                st.error(actuator)
            else:
                st.success(actuator)

            st.markdown("---")

            st.subheader("📊 Time-Series Data")
            st.dataframe(pd.DataFrame(st.session_state.logs).tail(5))

        time.sleep(1)

    # =========================
    # FINAL EXPLANATION
    # =========================
    st.markdown("---")

    st.subheader("📘 Explanation (As per AIoT Flow)")

    st.markdown("""
**📡 Sensor** → Collects real-world data  
**⚡ Edge Processing** → Applies filters/rules  
**📶 Gateway/Network** → Transfers data  
**🚚 Ingestion/Stream** → Streams data (Kafka-like)  
**💾 Storage** → Stores time-series data  
**🧠 Model Serving** → AI detects patterns/anomalies  
**⚙️ Decision (API)** → Generates action decision  
**🔔 Actuator/Alert** → Executes action automatically  
""")

    st.info("""
🔄 Flow:
Sensor → Edge → Gateway → Ingestion → Storage → Model → Decision → Actuator
""")
    
elif demo == "📊 Batch vs Streaming":

    import streamlit as st
    import random
    import time

    st.markdown("## 📊 Batch vs Streaming Processing Demo")

    st.write("Compare how data is processed in Batch vs Streaming")

    st.markdown("---")

    # =========================
    # STATE
    # =========================
    if "batch_data" not in st.session_state:
        st.session_state.batch_data = []

    if "stream_data" not in st.session_state:
        st.session_state.stream_data = []

    if "batch_result" not in st.session_state:
        st.session_state.batch_result = []

    if "running" not in st.session_state:
        st.session_state.running = False

    # =========================
    # CONTROLS
    # =========================
    col1, col2 = st.columns(2)

    if col1.button("▶ Start"):
        st.session_state.running = True

    if col2.button("⏹ Stop"):
        st.session_state.running = False

    placeholder = st.empty()

    # =========================
    # LOOP
    # =========================
    while st.session_state.running:

        temp = random.randint(20, 90)

        # Add data
        st.session_state.batch_data.append(temp)
        st.session_state.stream_data.append(temp)

        # =========================
        # STREAMING (REAL-TIME)
        # =========================
        if temp > 70:
            stream_result = "⚠️ Alert"
        else:
            stream_result = "✅ Normal"

        # =========================
        # BATCH (ONLY AFTER 10 VALUES)
        # =========================
        batch_result = None  # reset every loop

        if len(st.session_state.batch_data) == 10:

            avg_temp = sum(st.session_state.batch_data) / 10

            if avg_temp > 70:
                batch_result = "⚠️ Alert (Batch)"
            else:
                batch_result = "✅ Normal (Batch)"

            st.session_state.batch_result.append(batch_result)

            # clear batch
            st.session_state.batch_data = []

        # =========================
        # UI
        # =========================
        with placeholder.container():

            colA, colB = st.columns(2)

            # STREAMING
            with colA:
                st.subheader("⚡ Streaming (Real-Time)")
                st.metric("Latest Temp", temp)
                st.write("Result:", stream_result)
                st.success("Processed instantly (Low latency)")

            # BATCH
            with colB:
                st.subheader("🗂️ Batch Processing (Size = 10)")

                progress = len(st.session_state.batch_data)

                st.progress(progress / 10)

                st.write("Collected Data:", st.session_state.batch_data)

                if batch_result:
                    st.warning("Batch Completed ✅")
                    st.success(f"Result: {batch_result}")
                else:
                    st.info(f"Waiting... ({progress}/10)")

            st.markdown("---")

            st.subheader("📊 Batch Results History")
            st.write(st.session_state.batch_result[-5:])

        time.sleep(1)


elif demo == "🧹 Data Cleaning Pipeline":

    import streamlit as st
    import pandas as pd

    st.markdown("## 🧹 Data Cleaning Pipeline (Step-by-Step Tables)")

    st.markdown("---")

    # =========================
    # STATE
    # =========================
    if "step" not in st.session_state:
        st.session_state.step = 0

    if "raw" not in st.session_state:
        st.session_state.raw = pd.DataFrame({
            "Temperature": [25, 27, None, 30, 999, 28, 26, None, 27, -20],
            "Humidity": [60, None, 58, 62, 65, None, 59, 61, 300, 57],
            "Pressure": [101, 102, 103, None, 5000, 101, 102, 103, None, 100]
        })

    # =========================
    # RAW DATA
    # =========================
    st.subheader("📊 Raw Data")
    st.dataframe(st.session_state.raw)

    st.markdown("---")

    # =========================
    # STEP 1: MISSING VALUES
    # =========================
    if st.session_state.step >= 1:
        step1 = st.session_state.raw.fillna(st.session_state.raw.mean())
        st.subheader("1️⃣ Missing Values Handled")
        st.dataframe(step1)

    if st.session_state.step == 0:
        if st.button("1️⃣ Handle Missing Values"):
            st.session_state.step = 1
            st.rerun()

    # =========================
    # STEP 2: OUTLIERS
    # =========================
    if st.session_state.step >= 2:
        step2 = step1[
            (step1["Temperature"] >= 0) & (step1["Temperature"] <= 100)
        ]
        step2 = step2[
            (step2["Humidity"] >= 0) & (step2["Humidity"] <= 100)
        ]
        step2 = step2[
            (step2["Pressure"] >= 90) & (step2["Pressure"] <= 110)
        ]

        st.subheader("2️⃣ Outliers Removed")
        st.dataframe(step2)

    if st.session_state.step == 1:
        if st.button("2️⃣ Remove Outliers"):
            st.session_state.step = 2
            st.rerun()

    # =========================
    # STEP 3: NOISE REDUCTION
    # =========================
    if st.session_state.step >= 3:
        step3 = step2.copy()
        step3["Temperature"] = step3["Temperature"].rolling(2).mean().bfill()

        st.subheader("3️⃣ Noise Reduced (Smoothing)")
        st.dataframe(step3)

    if st.session_state.step == 2:
        if st.button("3️⃣ Noise Reduction"):
            st.session_state.step = 3
            st.rerun()

    # =========================
    # STEP 4: ERROR CORRECTION
    # =========================
    if st.session_state.step >= 4:
        step4 = step3.copy()
        step4["Humidity"] = step4["Humidity"].apply(
            lambda x: min(max(x, 0), 100)
        )

        st.subheader("4️⃣ Error Corrected")
        st.dataframe(step4)

    if st.session_state.step == 3:
        if st.button("4️⃣ Error Correction"):
            st.session_state.step = 4
            st.rerun()

    # =========================
    # STEP 5: DATA REDUCTION
    # =========================
    if st.session_state.step >= 5:
        step5 = step4.copy()

        before = len(step5)
        step5 = step5.drop_duplicates()
        after = len(step5)

        st.subheader("5️⃣ Data Reduction")
        st.dataframe(step5)
        st.info(f"Removed {before - after} duplicate rows")

        st.markdown("""
👉 Removes redundant/duplicate data  
👉 Saves storage and improves efficiency  
""")

    if st.session_state.step == 4:
        if st.button("5️⃣ Data Reduction"):
            st.session_state.step = 5
            st.rerun()

    # =========================
    # STEP 6: STANDARDIZATION
    # =========================
    if st.session_state.step >= 6:
        step6 = (step5 - step5.mean()) / step5.std(ddof=0)

        st.subheader("6️⃣ Standardized Data (Mean = 0, Std = 1)")
        st.dataframe(step6)

        st.markdown("### 📊 Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Mean (≈ 0):")
            st.write(step6.mean())

        with col2:
            st.write("Std Dev (≈ 1):")
            st.write(step6.std(ddof=0))

        st.info("Note: Small dataset → values may be close to 0 and 1, not exact.")

        st.success("✅ Data is fully cleaned and ML-ready")

    if st.session_state.step == 5:
        if st.button("6️⃣ Standardization"):
            st.session_state.step = 6
            st.rerun()

    # =========================
    # RESET
    # =========================
    st.markdown("---")

    if st.button("🔄 Reset"):
        st.session_state.step = 0
        st.rerun()

    # =========================
    # FLOW
    # =========================
    st.info("""
🔄 Pipeline Flow:
Raw → Missing → Outliers → Noise → Error → Reduction → Standardization
""")


elif demo == "🔧 Predict Equipment Failure":

    import streamlit as st
    import pandas as pd
    import numpy as np
    import random
    import time
    from sklearn.linear_model import LogisticRegression

    st.markdown("## 🔧 Predict Equipment Failure (Real-World AIoT)")

    st.write("Supervised Learning + Realistic Machine Behavior")

    st.markdown("---")

    # =========================
    # TRAINING DATA
    # =========================
    train_data = pd.DataFrame({
        "Temperature": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "Vibration":  [20, 25, 30, 35, 40, 45, 50, 60, 70, 80],
        "Label":      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })

    X = train_data[["Temperature", "Vibration"]]
    y = train_data["Label"]

    model = LogisticRegression()
    model.fit(X, y)

    st.success("✅ Model Trained on Historical Data")

    # =========================
    # STATE
    # =========================
    if "data" not in st.session_state:
        st.session_state.data = []

    if "running" not in st.session_state:
        st.session_state.running = False

    if "drift" not in st.session_state:
        st.session_state.drift = 0

    col1, col2, col3 = st.columns(3)

    if col1.button("▶ Start Monitoring"):
        st.session_state.running = True

    if col2.button("⏹ Stop"):
        st.session_state.running = False

    if col3.button("🔧 Maintenance Reset"):
        st.session_state.drift = 0
        st.success("Machine Serviced → Back to Healthy State")

    placeholder = st.empty()

    # =========================
    # LOOP
    # =========================
    while st.session_state.running:

        # Gradual degradation
        st.session_state.drift += random.uniform(0, 0.4)

        # Mostly normal, sometimes risky
        if random.random() < 0.9:
            temp = int(50 + st.session_state.drift + random.randint(-5, 5))
            vibration = int(20 + st.session_state.drift + random.randint(-5, 5))
        else:
            temp = int(75 + st.session_state.drift + random.randint(0, 10))
            vibration = int(50 + st.session_state.drift + random.randint(0, 10))

        input_data = np.array([[temp, vibration]])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        failure_percent = round(prob * 100, 2)

        if prediction == 1:
            result = "⚠️ Failure Likely"
        else:
            result = "✅ Healthy"

        # Store data
        st.session_state.data.append({
            "Temperature": temp,
            "Vibration": vibration,
            "Failure %": failure_percent,
            "Prediction": result
        })

        # =========================
        # UI
        # =========================
        with placeholder.container():

            colA, colB = st.columns(2)

            with colA:
                st.subheader("📡 Sensor Data")
                st.metric("Temperature", temp)
                st.metric("Vibration", vibration)

            with colB:
                st.subheader("🧠 AI Prediction")
                st.write(result)

                st.write(f"Failure Probability: {failure_percent}%")
                st.progress(prob)

                if failure_percent > 70:
                    st.error("🔴 High Risk - Immediate Action Required")
                elif failure_percent > 40:
                    st.warning("🟠 Moderate Risk - Monitor Closely")
                else:
                    st.success("🟢 Low Risk - Normal Operation")

            st.markdown("---")

            st.subheader("📊 Monitoring History")
            st.dataframe(pd.DataFrame(st.session_state.data).tail(6))

        time.sleep(1)

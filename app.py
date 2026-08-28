import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DiabCare AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* ==========================================================
   HEADER
   ========================================================== */

.main-header {
    background: linear-gradient(135deg, #162447 0%, #284b9b 55%, #4267b2 100%);
    padding: 35px 40px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(30, 60, 120, 0.18);
}

.main-header h1 {
    font-size: 44px;
    margin: 0 0 8px 0;
    font-weight: 800;
    color: white;
}

.main-header p {
    font-size: 19px;
    margin: 0;
    color: #e8efff;
}

.header-badge {
    display: inline-block;
    margin-top: 20px;
    padding: 7px 14px;
    border-radius: 20px;
    background: rgba(255,255,255,0.14);
    color: white;
    font-size: 13px;
    font-weight: 600;
}


/* ==========================================================
   CARDS
   ========================================================== */

.card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #e4e9f2;
    box-shadow: 0 5px 20px rgba(15, 35, 70, 0.06);
    margin-bottom: 22px;
}

.card-title {
    font-size: 21px;
    font-weight: 750;
    color: #182848;
    margin-bottom: 8px;
}

.card-subtitle {
    color: #64748b;
    font-size: 14px;
}


/* ==========================================================
   UPLOAD AREA
   ========================================================== */

.upload-card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    border: 2px dashed #cbd5e1;
    box-shadow: 0 5px 20px rgba(15, 35, 70, 0.05);
}

.upload-title {
    font-size: 23px;
    font-weight: 750;
    color: #182848;
}

.upload-description {
    color: #64748b;
    margin-bottom: 18px;
}


/* ==========================================================
   RESULT CARDS
   ========================================================== */

.result-high {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    border-left: 7px solid #dc2626;
    padding: 26px;
    border-radius: 16px;
    margin-top: 15px;
}

.result-low {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    border-left: 7px solid #16a34a;
    padding: 26px;
    border-radius: 16px;
    margin-top: 15px;
}

.result-uncertain {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 7px solid #f59e0b;
    padding: 26px;
    border-radius: 16px;
    margin-top: 15px;
}

.result-high h2,
.result-low h2,
.result-uncertain h2 {
    margin-top: 0;
}


/* ==========================================================
   METRICS
   ========================================================== */

.metric-box {
    background: white;
    padding: 22px 15px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(15, 35, 70, 0.05);
    min-height: 110px;
}

.metric-title {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #182848;
    margin-top: 8px;
}


/* ==========================================================
   RISK BADGES
   ========================================================== */

.risk-high {
    display: inline-block;
    background: #dc2626;
    color: white;
    padding: 7px 20px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 14px;
}

.risk-low {
    display: inline-block;
    background: #16a34a;
    color: white;
    padding: 7px 20px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 14px;
}

.risk-uncertain {
    display: inline-block;
    background: #f59e0b;
    color: white;
    padding: 7px 20px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 14px;
}


/* ==========================================================
   MODEL STATUS
   ========================================================== */

.status-online {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    color: #166534;
    padding: 10px 14px;
    border-radius: 10px;
    font-weight: 600;
    text-align: center;
}

.status-offline {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    color: #991b1b;
    padding: 10px 14px;
    border-radius: 10px;
    font-weight: 600;
    text-align: center;
}


/* ==========================================================
   INFORMATION BOX
   ========================================================== */

.info-box {
    background: #eff6ff;
    border-left: 5px solid #3b82f6;
    padding: 18px 20px;
    border-radius: 12px;
    color: #1e3a8a;
    margin: 15px 0;
}

.warning-box {
    background: #fffbeb;
    border-left: 5px solid #f59e0b;
    padding: 18px 20px;
    border-radius: 12px;
    color: #78350f;
    margin: 15px 0;
}


/* ==========================================================
   SECTION HEADINGS
   ========================================================== */

.section-heading {
    font-size: 25px;
    font-weight: 800;
    color: #182848;
    margin-top: 28px;
    margin-bottom: 15px;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding-top: 30px;
    margin-top: 45px;
    border-top: 1px solid #dfe5ee;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

.sidebar-brand {
    font-size: 25px;
    font-weight: 800;
    color: #182848;
}

.sidebar-section {
    font-size: 17px;
    font-weight: 750;
    color: #182848;
    margin-top: 20px;
}


/* ==========================================================
   BUTTON
   ========================================================== */

.stButton > button {
    border-radius: 12px;
    font-weight: 750;
    min-height: 50px;
}


/* ==========================================================
   IMAGE
   ========================================================== */

img {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best.pt"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()
    model_status = True
    model_error = None

except Exception as e:
    model_status = False
    model_error = str(e)
    model = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🩺 DiabCare AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-section">About the System</div>',
        unsafe_allow_html=True
    )

    st.write(
        "DiabCare AI is an AI-powered screening prototype "
        "designed to identify visual patterns associated "
        "with diabetic foot ulcers."
    )

    st.markdown(
        '<div class="sidebar-section">How It Works</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
**01 — Upload**  
Upload a clear photograph of the foot.

**02 — Analyze**  
The trained computer-vision model processes the image.

**03 — Screen**  
The AI estimates the most likely visual class.

**04 — Assess**  
The result is translated into a screening risk level.

**05 — Act**  
The system provides an appropriate next-step recommendation.
""")

    st.markdown("---")

    if model_status:

        st.markdown(
            '<div class="status-online">● AI MODEL ONLINE</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="status-offline">● AI MODEL ERROR</div>',
            unsafe_allow_html=True
        )

        st.caption(model_error)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="main-header">
<h1>🩺 DiabCare AI</h1>
<p>AI-Powered Diabetic Foot Early Warning System</p>
<span class="header-badge">COMPUTER VISION • AI SCREENING • EARLY WARNING</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown("""
<div class="card">
<div class="card-title">Early detection can make a difference.</div>

<p>
Diabetic foot complications can develop gradually and may sometimes
go unnoticed. DiabCare AI uses computer vision to screen photographs
for visual patterns associated with diabetic foot ulcers.
</p>

<div class="info-box">
<b>How DiabCare AI helps:</b><br>
The system provides a rapid AI-based screening indication that can
support awareness and encourage timely professional evaluation.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">📷 Foot Image Analysis</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.25, 0.85])


# ============================================================
# LEFT — IMAGE UPLOAD
# ============================================================

with left:

    st.markdown("""
<div class="upload-card">
<div class="upload-title">Upload Foot Image</div>
<div class="upload-description">
Choose a clear photograph for AI screening.
</div>
</div>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear, well-lit photograph with the foot visible."
    )

    if uploaded_file is not None:

        try:

            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded Foot Image",
                use_container_width=True
            )

        except Exception:

            st.error(
                "The uploaded image could not be read. "
                "Please upload a valid JPG or PNG image."
            )

            image = None

    else:

        image = None


# ============================================================
# RIGHT — IMAGE GUIDELINES
# ============================================================

with right:

    st.markdown("""
<div class="card">

<div class="card-title">📋 Image Guidelines</div>

<p class="card-subtitle">
For better screening results:
</p>

<p>
✅ Use a clear photograph<br>
✅ Ensure good lighting<br>
✅ Keep the foot fully visible<br>
✅ Avoid excessive blur<br>
✅ Avoid heavy filters<br>
✅ Focus on the affected area if visible
</p>

<div class="warning-box">
<b>Important:</b><br>
AI screening is experimental and does not replace
professional medical assessment.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if image is not None:

    st.markdown("")

    analyze = st.button(
        "🔍  ANALYZE FOOT IMAGE",
        use_container_width=True,
        type="primary"
    )

    if analyze:

        if not model_status:

            st.error(
                "The AI model could not be loaded. "
                "Please verify that best.pt exists."
            )

        else:

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            with st.spinner(
                "🧠 DiabCare AI is analyzing the image..."
            ):

                try:

                    results = model.predict(
                        source=image,
                        verbose=False
                    )

                    result = results[0]

                    probabilities = result.probs

                    top_class = int(probabilities.top1)

                    confidence = float(
                        probabilities.top1conf
                    )

                    class_name = model.names[top_class]

                    all_probs = probabilities.data.tolist()

                    prediction_success = True

                except Exception as e:

                    prediction_success = False
                    prediction_error = str(e)


            # ------------------------------------------------
            # HANDLE PREDICTION ERROR
            # ------------------------------------------------

            if not prediction_success:

                st.error(
                    "The image could not be analyzed."
                )

                st.caption(prediction_error)

            else:

                # ------------------------------------------------
                # CALCULATE SCREENING STATUS
                # ------------------------------------------------

                confidence_percent = confidence * 100

                is_ulcer = (
                    class_name.lower().strip() == "ulcer"
                )

                # Confidence threshold
                if confidence_percent < 70:

                    screening_status = "UNCERTAIN"
                    risk_level = "UNCERTAIN"

                elif is_ulcer:

                    screening_status = "POSSIBLE ULCER"
                    risk_level = "HIGH"

                else:

                    screening_status = "NO OBVIOUS ULCER"
                    risk_level = "LOW"


                # =================================================
                # RESULTS HEADER
                # =================================================

                st.markdown(
                    '<div class="section-heading">🧠 AI Screening Result</div>',
                    unsafe_allow_html=True
                )


                # =================================================
                # RESULT CARD
                # =================================================

                if risk_level == "HIGH":

                    st.markdown("""
<div class="result-high">
<h2>⚠️ Possible Ulcer Pattern Detected</h2>
<p>
The AI identified visual patterns associated with a
possible diabetic foot ulcer.
</p>
</div>
""", unsafe_allow_html=True)

                elif risk_level == "LOW":

                    st.markdown("""
<div class="result-low">
<h2>✅ No Obvious Ulcer Pattern Detected</h2>
<p>
The AI did not identify a strong visual pattern
associated with an ulcer in this image.
</p>
</div>
""", unsafe_allow_html=True)

                else:

                    st.markdown("""
<div class="result-uncertain">
<h2>🟡 Inconclusive Screening Result</h2>
<p>
The AI confidence is not high enough to provide a
strong screening indication. Consider retaking the
image under better conditions.
</p>
</div>
""", unsafe_allow_html=True)


                # =================================================
                # SCREENING SUMMARY
                # =================================================

                st.markdown(
                    '<div class="section-heading">📊 Screening Summary</div>',
                    unsafe_allow_html=True
                )

                col1, col2, col3 = st.columns(3)


                # Prediction
                with col1:

                    st.markdown(
                        f"""
<div class="metric-box">
<div class="metric-title">AI Prediction</div>
<div class="metric-value">{class_name.upper()}</div>
</div>
""",
                        unsafe_allow_html=True
                    )


                # Confidence
                with col2:

                    st.markdown(
                        f"""
<div class="metric-box">
<div class="metric-title">AI Confidence</div>
<div class="metric-value">{confidence_percent:.1f}%</div>
</div>
""",
                        unsafe_allow_html=True
                    )


                # Risk
                with col3:

                    if risk_level == "HIGH":

                        badge = '<span class="risk-high">HIGH</span>'

                    elif risk_level == "LOW":

                        badge = '<span class="risk-low">LOW</span>'

                    else:

                        badge = '<span class="risk-uncertain">UNCERTAIN</span>'


                    st.markdown(
                        f"""
<div class="metric-box">
<div class="metric-title">Screening Risk</div>
<br>
{badge}
</div>
""",
                        unsafe_allow_html=True
                    )


                # =================================================
                # PROBABILITIES
                # =================================================

                st.markdown(
                    '<div class="section-heading">📈 AI Confidence Breakdown</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                for i, prob in enumerate(all_probs):

                    name = model.names[i]

                    probability = float(prob)

                    probability_percent = probability * 100

                    st.write(
                        f"**{name.capitalize()}** — "
                        f"{probability_percent:.1f}%"
                    )

                    st.progress(
                        min(max(probability, 0.0), 1.0)
                    )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


                # =================================================
                # RECOMMENDATION
                # =================================================

                st.markdown(
                    '<div class="section-heading">💡 Recommended Action</div>',
                    unsafe_allow_html=True
                )


                if risk_level == "HIGH":

                    st.error(
                        "Professional medical evaluation is recommended. "
                        "The AI result indicates a possible ulcer pattern "
                        "and should be reviewed by a qualified healthcare "
                        "professional."
                    )

                elif risk_level == "LOW":

                    st.success(
                        "Continue regular foot monitoring. "
                        "No obvious ulcer pattern was detected in this "
                        "image. A negative screening result does not "
                        "guarantee that the foot is completely healthy."
                    )

                else:

                    st.warning(
                        "Retake the photograph using better lighting, "
                        "a clear view of the foot, and minimal blur. "
                        "If there are concerning symptoms or visible "
                        "changes, seek professional medical advice."
                    )


                # =================================================
                # AI INTERPRETATION
                # =================================================

                st.markdown(
                    '<div class="section-heading">🧠 AI Interpretation</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
<div class="card">

<b>Screening status:</b> {screening_status}<br><br>

<b>Model confidence:</b> {confidence_percent:.1f}%<br><br>

<b>Interpretation:</b><br>
The AI classification represents the model's assessment
of visual patterns in the uploaded photograph. It should
be used as a screening aid rather than a medical diagnosis.

</div>
""",
                    unsafe_allow_html=True
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

<b>🩺 DiabCare AI</b><br>
AI-powered diabetic foot screening prototype

<br><br>

⚠️ <b>Medical Disclaimer:</b><br>
This system is intended for research, educational and
screening purposes only. It does not provide a medical
diagnosis and does not replace professional healthcare
evaluation.

</div>
""", unsafe_allow_html=True)

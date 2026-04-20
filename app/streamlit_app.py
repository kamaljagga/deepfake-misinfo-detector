import os
import sys
import tempfile
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from module_a_deepfake.face_detector import FaceDetector
from module_a_deepfake.classifier import DeepfakeClassifier
from module_a_deepfake.aggregator import aggregate_frame_predictions
from module_a_deepfake.frame_extractor import extract_frames
from module_b_misinfo.model import MisinfoDetector

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Load models once ───────────────────────────────────────────
@st.cache_resource
def load_models():
    deepfake_path = os.path.join(
        os.path.dirname(__file__), '..', 'models',
        'deepfake', 'efficientnet_b0.pth'
    )
    misinfo_path = os.path.join(
        os.path.dirname(__file__), '..', 'models',
        'nlp', 'distilbert_finetuned.pth'
    )
    face_det     = FaceDetector()
    deepfake_clf = DeepfakeClassifier(model_path=deepfake_path)
    misinfo_clf  = MisinfoDetector(model_path=misinfo_path)
    return face_det, deepfake_clf, misinfo_clf

# ── Header ─────────────────────────────────────────────────────
st.title("🔍 Deepfake & Misinformation Detector")
st.caption("Fully offline · No cloud APIs · Built with PyTorch + DistilBERT")
st.divider()

# ── Load models with spinner ───────────────────────────────────
with st.spinner("Loading AI models..."):
    try:
        face_det, deepfake_clf, misinfo_clf = load_models()
        st.success("Models loaded successfully!", icon="✅")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

st.divider()

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎥 Deepfake Detector", "📰 Misinformation Detector"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DEEPFAKE DETECTOR
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload a video or image to check for deepfakes")

    uploaded = st.file_uploader(
        "Choose a file",
        type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
        help="Upload a video or image file"
    )

    if uploaded is not None:
        # Show preview for images
        if uploaded.type.startswith("image"):
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", width=300)

        if st.button("🔍 Analyze for Deepfakes", type="primary"):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save uploaded file
                file_path = os.path.join(tmpdir, uploaded.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                with st.spinner("Extracting frames and detecting faces..."):
                    frames = extract_frames(
                        file_path,
                        os.path.join(tmpdir, "frames"),
                        fps=2
                    )

                if not frames:
                    st.error("Could not extract frames from the file.")
                else:
                    with st.spinner(f"Analyzing {len(frames)} frames..."):
                        preds = []
                        bar   = st.progress(0)
                        for i, frame in enumerate(frames):
                            face = face_det.extract_face(frame)
                            pred = deepfake_clf.predict(face)
                            preds.append(pred)
                            bar.progress((i + 1) / len(frames))

                    result = aggregate_frame_predictions(preds)

                    st.divider()
                    # ── Verdict ──
                    # CHANGE TO — show uncertain when confidence is low:
                    confidence = result["confidence"]
                    if result["verdict"] == "FALSE" and confidence > 0.70:
                        st.error("🔴 LIKELY MISINFORMATION", icon="🚨")
                    elif result["verdict"] == "TRUE" and confidence > 0.70:
                        st.success("🟢 APPEARS CREDIBLE", icon="✅")
                    else:
                        st.warning(
                            f"⚠️ UNCERTAIN — model confidence too low ({confidence*100:.1f}%)\n\n"
                            "Please verify this information from trusted sources.",
                            icon="⚠️"
                        )

                    # ── Metrics ──
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Fake Frame Ratio",
                        f"{result['fake_ratio']*100:.1f}%"
                    )
                    col2.metric(
                        "Frames Analyzed",
                        result["frames_analyzed"]
                    )
                    col3.metric(
                        "Avg Confidence",
                        f"{result['avg_confidence']*100:.1f}%"
                    )

                    st.divider()
                    with st.expander("Detailed frame results"):
                        for i, p in enumerate(preds):
                            icon = "🔴" if p["label"] == "FAKE" else "🟢"
                            st.write(
                                f"Frame {i+1}: {icon} {p['label']} "
                                f"({p['confidence']*100:.1f}%)"
                            )

# ══════════════════════════════════════════════════════════════
# TAB 2 — MISINFORMATION DETECTOR
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Paste a news headline or article to check credibility")

    text_input = st.text_area(
        "Enter text to analyze",
        height=150,
        placeholder="Paste a news headline or article excerpt here..."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_btn = st.button(
            "🔍 Analyze Text",
            type="primary",
            use_container_width=True
        )

    if analyze_btn:
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Running NLP analysis..."):
                result = misinfo_clf.predict(text_input)

            st.divider()
            # ── Verdict ──
            if result["verdict"] == "FALSE":
                st.error("🔴 LIKELY MISINFORMATION", icon="🚨")
            else:
                st.success("🟢 APPEARS CREDIBLE", icon="✅")

            # ── Scores ──
            col1, col2 = st.columns(2)
            col1.metric(
                "Confidence",
                f"{result['confidence']*100:.1f}%"
            )
            col2.metric(
                "Misinfo Score",
                f"{result['scores']['false']*100:.1f}%"
            )

            st.divider()
            # ── Visual bar ──
            st.write("Credibility breakdown:")
            st.progress(
                result["scores"]["true"],
                text=f"Credible: {result['scores']['true']*100:.1f}%"
            )
            st.progress(
                result["scores"]["false"],
                text=f"Misinformation: {result['scores']['false']*100:.1f}%"
            )

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption(
    "Final Year Project · Deepfake & Misinformation Detection · "
    "Built with PyTorch, HuggingFace Transformers, OpenCV"
)
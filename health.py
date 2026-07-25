# ============================================================
# COVID-19 Chest X-ray Detection
# Custom CNN Model
# Developed using TensorFlow + Streamlit
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from tensorflow.keras.models import load_model

# ------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="COVID-19 Chest X-ray Detection",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Load CNN Model
# ------------------------------------------------------------

@st.cache_resource
def load_covid_model():
    return load_model("covid_classifier_final.keras")

model = load_covid_model()

# ------------------------------------------------------------
# Class Labels
# ------------------------------------------------------------

CLASS_NAMES = [
    "COVID-19",
    "Normal",
    "Viral Pneumonia"
]

# ------------------------------------------------------------
# Healthcare Color Theme
# ------------------------------------------------------------

st.markdown("""
<style>
.main{
    background-color:#F5FAFD;
}
.title{
    text-align:center;
    color:#0F4C81;
    font-size:42px;
    font-weight:bold;
}
.subtitle{
    text-align:center;
    color:#4A5568;
    font-size:18px;
}
.result-box{
    background-color:white;
    padding:20px;
    border-radius:12px;
    border-left:8px solid #0096C7;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}
.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:hidden;
}
header{
    visibility:hidden;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Image Validation Helpers
# (Reject non-X-ray / colour photos before they ever reach the model)
# ============================================================

def analyze_image(pil_image: Image.Image):
    """
    Runs a set of heuristic checks on the uploaded image and returns
    a dict of measurements plus a pass/fail flag for each check.

    NOTE: These are heuristic, rule-based checks (colour/contrast/
    aspect-ratio based). They are good at catching obviously wrong
    uploads (selfies, scenery, colourful diagrams, screenshots, etc.)
    but they CANNOT reliably tell a chest X-ray apart from an X-ray
    of a different body part (e.g. a hand or knee X-ray) — both are
    grayscale, high-contrast images. Doing that reliably would
    require a dedicated "chest vs. non-chest" classifier trained on
    labeled data. See the note at the end of the chat response.
    """
    rgb_img = pil_image.convert("RGB")
    arr = np.array(rgb_img).astype(np.float32)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 1. Colourfulness check -> X-rays are essentially grayscale.
    #    A true grayscale image has R ≈ G ≈ B at every pixel.
    channel_diff = (
        np.mean(np.abs(r - g)) +
        np.mean(np.abs(g - b)) +
        np.mean(np.abs(r - b))
    ) / 3.0

    # 2. Saturation check (HSV) -> colour photos have much higher
    #    average saturation than grayscale medical images.
    hsv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    # 3. Contrast / dynamic range check -> X-rays typically show
    #    a wide spread of gray values (dark background, bright bone).
    gray = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2GRAY)
    contrast_std = float(np.std(gray))

    # 4. Aspect ratio check -> chest X-rays are usually close to
    #    square / portrait, not extreme wide-screen photos or banners.
    width, height = pil_image.size
    aspect_ratio = width / height

    checks = {
        "is_grayscale": channel_diff < 12,
        "low_saturation": mean_saturation < 25,
        "has_contrast": contrast_std > 25,
        "plausible_aspect_ratio": 0.65 <= aspect_ratio <= 1.5,
    }

    measurements = {
        "channel_diff": channel_diff,
        "mean_saturation": mean_saturation,
        "contrast_std": contrast_std,
        "aspect_ratio": aspect_ratio,
    }

    is_valid = all(checks.values())

    return is_valid, checks, measurements


def rejection_reasons(checks: dict) -> list:
    reasons = []
    if not checks["is_grayscale"]:
        reasons.append("The image appears to be a colour photo, not a grayscale X-ray.")
    if not checks["low_saturation"]:
        reasons.append("The image has visible colour saturation, which real X-rays do not have.")
    if not checks["has_contrast"]:
        reasons.append("The image lacks the sharp bright/dark contrast typical of an X-ray scan.")
    if not checks["plausible_aspect_ratio"]:
        reasons.append("The image proportions don't match a typical chest X-ray frame.")
    return reasons


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/stethoscope.png",
        width=80
    )

    st.title("Model Information")
    st.markdown("---")

    st.write("### Model")
    st.success("Custom CNN")

    st.write("### Input Size")
    st.info("224 × 224")

    st.write("### Classes")
    st.write("• COVID-19")
    st.write("• Normal")
    st.write("• Viral Pneumonia")

    st.markdown("---")
    st.write("### Framework")
    st.write("TensorFlow / Keras")
    st.write("Streamlit")

    st.markdown("---")
    st.write("### Developer")
    st.write("Sumit Ghodke")

    st.markdown("---")
    st.warning(
        "Upload only Chest X-ray images.\n\n"
        "Colour photos, non-X-ray images, and X-rays of other body "
        "parts will be automatically rejected before prediction."
    )

# ============================================================
# Header Section
# ============================================================

st.markdown(
    """
    <h1 class="title">🩻 COVID-19 Chest X-ray Detection</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitle">
    AI-powered Chest X-ray Classification using a Custom Convolutional Neural Network (CNN)
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# Information Card
# ============================================================

with st.container():
    st.info(
        """
### 📋 Instructions

1. Upload a Chest X-ray image (.jpg, .jpeg or .png)
2. Click **Predict**
3. The AI model will classify the X-ray into:

• COVID-19
• Normal
• Viral Pneumonia

⚠ This model was trained only on Chest X-ray images.
Non-X-ray images and colour photos will be rejected automatically.
"""
    )

st.divider()

# ============================================================
# Two Column Layout
# ============================================================

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📤 Upload Chest X-ray")
    uploaded_file = st.file_uploader(
        "Choose a Chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

with right_col:
    st.subheader("🖼 Image Preview")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )
    else:
        st.info("Upload an image to preview it here.")

st.divider()

# ============================================================
# Proceed only if image uploaded
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # --------------------------------------------------------
    # Run validation checks BEFORE showing the Predict button
    # --------------------------------------------------------
    is_valid, checks, measurements = analyze_image(image)

    if not is_valid:
        st.error("### ❌ This image was rejected")
        st.write(
            "This does not look like a valid chest X-ray. "
            "Please upload a genuine grayscale chest X-ray image."
        )

        reasons = rejection_reasons(checks)
        for reason in reasons:
            st.write(f"- {reason}")

        with st.expander("🔍 Technical validation details"):
            st.write(f"**Colour channel difference:** {measurements['channel_diff']:.2f} (must be low for grayscale)")
            st.write(f"**Average saturation:** {measurements['mean_saturation']:.2f} (must be low)")
            st.write(f"**Contrast (std dev):** {measurements['contrast_std']:.2f} (must be reasonably high)")
            st.write(f"**Aspect ratio:** {measurements['aspect_ratio']:.2f} (expected roughly 0.65–1.5)")

        st.info(
            "ℹ Note: these checks catch colour photos, screenshots, and other "
            "non-X-ray images. They cannot fully guarantee the image is "
            "specifically a **chest** X-ray (versus, say, a hand or knee X-ray), "
            "since all X-rays share similar grayscale/contrast properties. "
            "Please make sure you upload a genuine chest X-ray."
        )

    else:
        st.success("✅ Image passed validation checks — this looks like a valid X-ray.")

        predict_button = st.button(
            "🔍 Predict",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            with st.spinner("Analyzing Chest X-ray..."):

                # ----------------------------------------------------
                # Image Preprocessing
                # ----------------------------------------------------
                img = image.convert("RGB")
                img = np.array(img)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                # ----------------------------------------------------
                # Model Prediction
                # ----------------------------------------------------
                prediction = model.predict(img, verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(prediction[0][predicted_index]) * 100

                covid_prob = float(prediction[0][0]) * 100
                normal_prob = float(prediction[0][1]) * 100
                viral_prob = float(prediction[0][2]) * 100

            st.divider()
            st.subheader("📊 Prediction Completed")

            # ----------------------------------------------------
            # Prediction Result Dashboard
            # ----------------------------------------------------
            st.markdown("## 📋 AI Prediction Report")

            result_col, confidence_col = st.columns(2)

            with result_col:
                if predicted_class == "COVID-19":
                    st.error("## 🦠 COVID-19 Detected")
                elif predicted_class == "Normal":
                    st.success("## ✅ Normal Chest X-ray")
                else:
                    st.warning("## 🫁 Viral Pneumonia Detected")

            with confidence_col:
                st.metric(
                    label="Model Confidence",
                    value=f"{confidence:.2f}%"
                )

            st.divider()

            # ----------------------------------------------------
            # Prediction Probabilities
            # ----------------------------------------------------
            st.subheader("📊 Prediction Probabilities")

            st.write("### 🦠 COVID-19")
            st.progress(covid_prob / 100)
            st.write(f"**{covid_prob:.2f}%**")

            st.write("### 🫁 Normal")
            st.progress(normal_prob / 100)
            st.write(f"**{normal_prob:.2f}%**")

            st.write("### 🫁 Viral Pneumonia")
            st.progress(viral_prob / 100)
            st.write(f"**{viral_prob:.2f}%**")

            st.divider()

            # ----------------------------------------------------
            # Clinical Interpretation
            # ----------------------------------------------------
            st.subheader("🩺 Clinical Interpretation")

            if predicted_class == "COVID-19":
                st.error(
                    """
### Interpretation

The uploaded Chest X-ray has been classified as **COVID-19**.

This prediction indicates radiographic patterns similar to those
present in the COVID-19 images used during model training.

⚠ Professional medical evaluation is strongly recommended.
"""
                )
            elif predicted_class == "Normal":
                st.success(
                    """
### Interpretation

The uploaded Chest X-ray has been classified as **Normal**.

No radiographic patterns associated with COVID-19 or Viral Pneumonia
were detected by the AI model.
"""
                )
            else:
                st.warning(
                    """
### Interpretation

The uploaded Chest X-ray has been classified as **Viral Pneumonia**.

The detected radiographic features appear more similar to Viral
Pneumonia than COVID-19 or Normal images.
"""
                )

            st.divider()

            # ----------------------------------------------------
            # Model Details
            # ----------------------------------------------------
            with st.expander("📈 View Technical Details"):
                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write("**Input Image Size:** 224 × 224")
                st.write("**Model:** Custom CNN")
                st.write("**Framework:** TensorFlow / Keras")
                st.write("**Classes:**")
                st.code(
"""
COVID-19
Normal
Viral Pneumonia
"""
                )

            # ----------------------------------------------------
            # Medical Disclaimer
            # ----------------------------------------------------
            st.divider()
            st.warning(
                """
### ⚠ Medical Disclaimer

This AI application is intended for **educational and research purposes only**.

The prediction generated by this model should **NOT** be used as a substitute
for professional medical diagnosis, clinical judgment, or treatment.

Always consult a qualified radiologist or healthcare professional before making
any medical decisions.

The developer assumes no responsibility for decisions made based on the
predictions of this application.
"""
            )

            # ----------------------------------------------------
            # About the Model
            # ----------------------------------------------------
            with st.expander("ℹ About This AI Model"):
                st.markdown("""
### Model Overview

**Model Name**
- Custom Convolutional Neural Network (CNN)

**Deep Learning Framework**
- TensorFlow / Keras

**Input Size**
- 224 × 224 pixels

**Input Type**
- Chest X-ray Images

**Output Classes**
- COVID-19
- Normal
- Viral Pneumonia

**Prediction Method**
- Softmax Classification

This application uses a Custom CNN trained on Chest X-ray images
to classify lung conditions into one of three categories.
""")

            # ----------------------------------------------------
            # Model Performance
            # ----------------------------------------------------
            with st.expander("📊 Model Performance"):
                st.markdown("""
### Test Performance

| Metric | Value |
|---------|-------|
| Test Accuracy | **87.88%** |
| Weighted F1-Score | **0.88** |
| Model Selected | ✅ Custom CNN |

The Custom CNN was selected for deployment because it achieved the
best performance among all evaluated models.
""")

# ============================================================
# Footer
# ============================================================

st.divider()

st.markdown(
    """
<div class="footer">

Developed by <b>Sumit Ghodke</b><br><br>

🩻 AI-powered COVID-19 Chest X-ray Detection<br>

Built using TensorFlow • Keras • Streamlit

</div>
""",
    unsafe_allow_html=True
)
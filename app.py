import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="🔢", layout="centered")

st.title("🔢 Handwritten Digit Recognizer")
st.write("Recognize handwritten digits using **K-Nearest Neighbors (KNN)** with feature standardization.")

@st.cache_resource
def train_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=3))
    ])
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return digits, model, accuracy

digits, model, accuracy = train_model()

st.success(f"Model ready • Test accuracy: {accuracy * 100:.2f}%")

st.subheader("🧪 Try a built-in handwritten sample")
sample_index = st.slider("Choose a sample", 0, len(digits.images) - 1, 0)

sample_image = digits.images[sample_index]
sample_features = digits.data[sample_index].reshape(1, -1)
sample_prediction = int(model.predict(sample_features)[0])
actual = int(digits.target[sample_index])

col1, col2 = st.columns(2)
with col1:
    st.image(sample_image, width=220, caption=f"Actual digit: {actual}")
with col2:
    st.metric("Predicted digit", sample_prediction)
    if sample_prediction == actual:
        st.success("Correct prediction ✓")
    else:
        st.warning("Prediction differs from the actual label.")

st.divider()

st.subheader("✍️ Upload your own handwritten digit")
st.caption("For best results, upload a clear image containing one centered handwritten digit on a plain background.")

uploaded = st.file_uploader("Upload PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"])


def image_to_features(image: Image.Image):
    image = image.convert("L")
    arr = np.array(image, dtype=np.uint8)

    # Determine whether the digit is dark-on-light or light-on-dark.
    if arr.mean() > 127:
        arr = 255 - arr

    # Remove very light background pixels and find the digit bounding box.
    mask = arr > 30
    if not mask.any():
        return None, None

    ys, xs = np.where(mask)
    cropped = Image.fromarray(arr).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    # Preserve aspect ratio and center the digit in a square canvas.
    w, h = cropped.size
    size = max(w, h)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(cropped, ((size - w) // 2, (size - h) // 2))

    # Match the scikit-learn digits dataset: 8x8, values scaled to 0..16.
    resized = canvas.resize((8, 8), Image.Resampling.LANCZOS)
    values = np.asarray(resized, dtype=np.float32) / 255.0 * 16.0
    return values, resized

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=260)

    features, processed = image_to_features(image)
    if features is None:
        st.error("No visible digit was detected. Please upload a clearer image.")
    else:
        prediction = int(model.predict(features.reshape(1, -1))[0])
        st.image(processed, caption="Processed 8×8 input", width=180)
        st.success(f"### Predicted digit: **{prediction}**")

        try:
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            confidence = float(np.max(probabilities)) * 100
            st.metric("Model confidence", f"{confidence:.2f}%")
        except Exception:
            pass

st.divider()

st.subheader("🤖 Model Information")
st.markdown(
    """
- **Dataset:** scikit-learn handwritten digits dataset
- **Image size:** 8 × 8 pixels
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Preprocessing:** StandardScaler
- **Neighbors:** 3
- **Language:** Python
- **Framework:** Streamlit
"""
)

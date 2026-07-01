import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.predict import SkinCancerPredictor
from src.utils.config import load_config
from src.utils.explain import generate_prediction_explanation
from src.utils.gradcam import GradCAM


st.set_page_config(
    page_title="Derma Vision AI",
    layout="wide",
)


@st.cache_resource
def load_predictor(checkpoint_path: str, model_key: str, config_path: str) -> SkinCancerPredictor:
    config = load_config(config_path)
    return SkinCancerPredictor(checkpoint_path, model_key, config)


def main() -> None:
    config_path = "configs/config.yaml"
    config = load_config(config_path)

    st.title("Derma Vision AI")
    st.markdown("<p style=\"font-size:18px; color:#666; margin-top:-12px;\">Skin cancer detection model</p>", unsafe_allow_html=True)
    st.caption("Upload a dermoscopic image to classify the lesion and inspect Grad-CAM evidence.")

    with st.sidebar:
        model_key = st.selectbox("Model", ["efficientnet", "vit"], index=0)
        checkpoint_default = f"checkpoints/best_{model_key}.pt"
        checkpoint_path = st.text_input("Checkpoint path", checkpoint_default)

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        st.warning(
            f"No checkpoint found at `{checkpoint}`. Train a model first, then refresh this app."
        )
        st.code(f"python -m src.training.train --model {model_key}", language="bash")
        return

    uploaded_file = st.file_uploader(
        "Upload lesion image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    predictor = load_predictor(str(checkpoint), model_key, config_path)
    prediction = predictor.predict(image)

    gradcam = GradCAM(predictor.model, model_key)
    try:
        gradcam_result = gradcam.generate(
            image=image,
            class_index=prediction["class_index"],
            image_size=config["data"]["image_size"],
            device=predictor.device,
        )
    finally:
        gradcam.close()

    left, right = st.columns(2)
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with right:
        st.image(gradcam_result.overlay, caption="Grad-CAM heatmap", use_container_width=True)

    class_name = prediction["class_name"]
    description = config["data"]["class_descriptions"].get(class_name, class_name)
    st.metric("Predicted class", f"{class_name} - {description}")
    st.metric("Confidence", f"{prediction['confidence'] * 100:.1f}%")

    st.subheader("Prediction Explanation")
    st.write(
        generate_prediction_explanation(
            predicted_class=class_name,
            confidence=prediction["confidence"],
            class_descriptions=config["data"]["class_descriptions"],
        )
    )

    st.subheader("Class probabilities")
    probabilities = {
        class_label: float(prob)
        for class_label, prob in zip(config["data"]["class_names"], prediction["probabilities"])
    }
    st.bar_chart(probabilities)


if __name__ == "__main__":
    main()



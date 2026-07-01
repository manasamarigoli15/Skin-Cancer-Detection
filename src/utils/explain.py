def generate_prediction_explanation(
    predicted_class: str,
    confidence: float,
    class_descriptions: dict[str, str],
) -> str:
    """Create a concise, deterministic explanation for the app."""
    description = class_descriptions.get(predicted_class, predicted_class)
    confidence_pct = confidence * 100.0
    caution = (
        "This result is a decision-support output, not a medical diagnosis. "
        "A dermatologist should review suspicious or changing lesions."
    )
    return (
        f"The model predicts {description} ({predicted_class}) with "
        f"{confidence_pct:.1f}% confidence. The heatmap highlights image regions "
        "that most influenced the score, usually areas with color, border, or texture "
        f"patterns learned during training. {caution}"
    )

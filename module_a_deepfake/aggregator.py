from collections import Counter


def aggregate_frame_predictions(predictions: list) -> dict:
    """
    Takes list of per-frame dicts and returns final verdict.
    predictions: [{"label": "FAKE", "confidence": 0.87}, ...]
    """
    if not predictions:
        return {
            "verdict":        "UNDETERMINED",
            "fake_ratio":     0.0,
            "avg_confidence": 0.0,
            "frames_analyzed": 0
        }

    valid = [p for p in predictions
             if p["label"] not in ["NO_FACE", "ERROR"]]

    if not valid:
        return {
            "verdict":         "UNDETERMINED",
            "fake_ratio":      0.0,
            "avg_confidence":  0.0,
            "frames_analyzed": 0
        }

    labels     = [p["label"] for p in valid]
    counts     = Counter(labels)
    fake_ratio = counts.get("FAKE", 0) / len(valid)
    avg_conf   = sum(p["confidence"] for p in valid) / len(valid)

    # Verdict: FAKE if more than 40% frames are fake
    verdict = "FAKE" if fake_ratio >= 0.4 else "REAL"

    return {
        "verdict":         verdict,
        "fake_ratio":      round(fake_ratio, 3),
        "avg_confidence":  round(avg_conf, 3),
        "frames_analyzed": len(valid),
        "fake_frames":     counts.get("FAKE", 0),
        "real_frames":     counts.get("REAL", 0)
    }
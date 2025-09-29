import json
from transformers import pipeline

# Load model once (so it doesn't reload for every request)
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)

def _none_payload():
    return {
        "scores": scores,
        "dominant_emotion": None,
        "recommendation": None,
    }

def _recommend_action(emotion: str):
    if emotion in ["anger", "disgust", "annoyance"]:
        return "Route case to Team Lead"
    elif emotion in ["sadness", "disappointment"]:
        return "Escalate to Senior Engineer with empathy training"
    elif emotion in ["joy", "excitement"]:
        return "Acknowledge positive feedback and thank customer"
    elif emotion == "neutral":
        return "Handle via standard support workflow"
    else:
        return "Review manually"

def emotion_detector(text_to_analyze: str):
    # Local blank guard (don’t even call API)
    if text_to_analyze is None or not str(text_to_analyze).strip():
        return _none_payload()

    try:
        results = emotion_pipeline(text_to_analyze)

        print(results) # debug: see raw HF output in your server console

        # Results is a list of lists -> [[{label:..., score:...}, ...]]
        scores = {item["label"].lower(): item["score"] for item in results[0]}

        #Dominant emotion
        dominant = max(scores, key=scores.get) if scores else None
        recommendation = _recommend_action(dominant) if dominant else None
       
        return {
            "scores": scores,
            "dominant_emotion": dominant,
            "recommendation": recommendation,
        }
    except Exception as e:
        print(f"Error during emotion detection: {e}")
        return _none_payload()
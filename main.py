from flask import Flask, request, jsonify
from transformers import pipeline
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict, Counter
import statistics
import os

# Step 1: Initialize Flask app
app = Flask(__name__)

# Step 2: Load Emotion & Risk Classifier Models (same public model for now)
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
risk_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

# Step 3: Crisis Keywords
crisis_keywords = {
    "hopeless", "worthless", "suicidal", "kill myself",
    "give up", "end it all", "can't go on", "no purpose",
    "don’t want to be here", "everything is meaningless",
    "better off without me", "I want to disappear",
    "I hate living", "I’m done with life", "I want it to end"
}

# Step 4: Initialize Firebase
cred = credentials.Certificate({
    "type": "service_account",
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
})

db = firestore.client()

# Utility: Format timestamp
def to_date(timestamp):
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    return timestamp.strftime('%Y-%m-%d')

# Step 5: Analyze Mood (Text)
@app.route('/analyzeMood', methods=['POST'])
def analyze_mood():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'user_id' not in data:
            return jsonify({"error": "Missing text or user_id"}), 400

        user_text = data['text']
        user_id = data['user_id']

        result = emotion_classifier(user_text)[0]
        emotion = result['label']
        confidence = round(result['score'], 2)

        keyword_crisis = any(word in user_text.lower() for word in crisis_keywords)
        risk_result = risk_classifier(user_text)[0]
        risk_label = risk_result['label'].lower()
        risk_score = risk_result['score']
        ai_crisis = risk_label in {"sadness", "anger", "fear"} and risk_score > 0.85
        crisis_detected = keyword_crisis or ai_crisis

        trigger_emergency = False
        emergency_settings = db.collection("user_emergency_settings").document(user_id).get()
        if emergency_settings.exists:
            user_settings = emergency_settings.to_dict()
            code_word = user_settings.get("code_word")
            if code_word and code_word in user_text.lower():
                trigger_emergency = True

        db.collection("mood_entries").add({
            "user_id": user_id,
            "text": user_text,
            "emotion": emotion,
            "confidence": confidence,
            "crisis": crisis_detected,
            "timestamp": datetime.now(),
            "risk_score": round(risk_score, 2),
            "risk_label": risk_label,
            "trigger_emergency": trigger_emergency
        })

        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
            "crisis": crisis_detected,
            "risk_score": round(risk_score, 2),
            "risk_label": risk_label,
            "trigger_emergency": trigger_emergency,
            "message": "Entry saved to Firebase successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Step 6: Set Emergency Code Word + Contact
@app.route('/setEmergency', methods=['POST'])
def set_emergency_code():
    try:
        data = request.get_json()
        if not data or 'user_id' not in data or 'code_word' not in data or 'emergency_contact' not in data:
            return jsonify({"error": "Missing fields"}), 400

        db.collection("user_emergency_settings").document(data['user_id']).set({
            "code_word": data['code_word'].lower(),
            "emergency_contact": data['emergency_contact'],
            "updated_at": datetime.now()
        })

        return jsonify({"message": "Emergency code and contact saved successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Step 7: Get Mood Trends
@app.route('/getMoodTrends', methods=['POST'])
def get_mood_trends():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        mood_entries = db.collection("mood_entries").where("user_id", "==", user_id).stream()
        grouped_data = defaultdict(list)

        for entry in mood_entries:
            doc = entry.to_dict()
            if "timestamp" not in doc:
                continue
            date_key = to_date(doc["timestamp"])
            grouped_data[date_key].append({
                "emotion": doc.get("emotion"),
                "risk_score": doc.get("risk_score", 0)
            })

        trends = {}
        for date, entries in grouped_data.items():
            risk_scores = [entry["risk_score"] for entry in entries]
            emotions = [entry["emotion"] for entry in entries if entry["emotion"]]

            average_risk = round(sum(risk_scores) / len(risk_scores), 2) if risk_scores else 0
            emotion_counts = dict(Counter(emotions))
            mood_swing = False
            if len(risk_scores) >= 2:
                std_dev = round(statistics.stdev(risk_scores), 2)
                mood_swing = std_dev > 2

            trends[date] = {
                "average_risk_score": average_risk,
                "emotion_distribution": emotion_counts,
                "mood_swing": mood_swing
            }

        return jsonify({
            "user_id": user_id,
            "trends": trends,
            "message": "Mood trends calculated successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Step 8: Log Visual Emotion (via camera)
@app.route('/logVisualEmotion', methods=['POST'])
def log_visual_emotion():
    try:
        data = request.get_json()
        if not data or 'user_id' not in data or 'emotion' not in data:
            return jsonify({"error": "Missing user_id or emotion"}), 400

        db.collection("visual_mood_entries").add({
            "user_id": data['user_id'],
            "emotion": data['emotion'],
            "confidence": data.get('confidence', None),
            "timestamp": datetime.now()
        })

        return jsonify({
            "message": "Visual emotion logged successfully",
            "user_id": data['user_id'],
            "emotion": data['emotion'],
            "confidence": data.get('confidence', None)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Step 9: Submit Daily Check-In (Slider + Tags + Optional Note)
@app.route('/submitCheckIn', methods=['POST'])
def submit_checkin():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        slider = data.get("slider_value")
        tags = data.get("tags", [])
        note = data.get("note", "")

        if not user_id or slider is None or not tags:
            return jsonify({"error": "Missing required fields"}), 400

        db.collection("user_checkins").document(user_id).collection("entries").add({
            "slider_value": slider,
            "tags": tags,
            "note": note,
            "timestamp": datetime.now()
        })

        return jsonify({"message": "Check-in saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Step 10: Run App
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host="0.0.0.0", port=port)

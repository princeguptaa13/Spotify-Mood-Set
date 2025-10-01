import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Emotion → Spotify playlist mapping
playlists = {
    "happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
    "sad": "https://open.spotify.com/playlist/37i9dQZF1DWVrtsSlLKzro",
    "angry": "https://open.spotify.com/playlist/37i9dQZF1DXd7j5DaScwJk",
    "neutral": "https://open.spotify.com/playlist/37i9dQZF1DX7KNKjOK0o75",
    "fear": "https://open.spotify.com/playlist/37i9dQZF1DWVIzZt2GAU4X",
    "surprise": "https://open.spotify.com/playlist/37i9dQZF1DX4fpCWaHOned",
    "disgust": "https://open.spotify.com/playlist/37i9dQZF1DX3WvGXE8FqYX"
}

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # decode base64 image
        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # resize for faster processing
        img = cv2.resize(img, (224, 224))

        # analyze emotion
        result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        emotion = result[0]["dominant_emotion"]

        # pick playlist
        playlist_url = playlists.get(emotion, playlists["neutral"])

        return jsonify({"emotion": emotion, "playlist": playlist_url})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import torch
import pandas as pd
from models.evolved_model import EvolvedNN, winner, config
from utils.extract_features import extract_features_from_audio
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Load NEAT model
neat_net = torch.load('models/evolved_neat_model.pth', map_location=torch.device('cpu'))
model = EvolvedNN(neat_net, winner)
model.load_state_dict(torch.load("models/evolved_neat_model.pth", map_location=torch.device('cpu')))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("audio")
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            features = extract_features_from_audio(file_path)
            input_tensor = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                output = model(input_tensor)
                pred_class = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).squeeze().tolist()
                prediction = "Autistic" if pred_class == 1 else "Non-Autistic"

                # Save results
                result_id = str(uuid.uuid4())[:8]
                result_file = os.path.join(app.config['RESULT_FOLDER'], f"report_{result_id}.csv")
                df = pd.DataFrame({
                    "Feature": ["f0", "Jitter", "Shimmer", "HNR"],
                    "Value": features
                })
                df["Prediction"] = prediction
                df["Confidence_NonAutistic"] = confidence[0]
                df["Confidence_Autistic"] = confidence[1]
                df.to_csv(result_file, index=False)

                return render_template("result.html", prediction=prediction, confidence=confidence,
                                       features=zip(["f0", "Jitter", "Shimmer", "HNR"], features),
                                       result_file=f"report_{result_id}.csv")
    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

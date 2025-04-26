import os
import json
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    session
)
from dotenv import load_dotenv
import openai

# ─── Load env & init Flask ─────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "please-change-me")

# ─── Solar Prediction / Dashboard Setup ────────────────────────────────────────
model = pickle.load(open("solar_energy_model.pkl", "rb"))
data  = pd.read_csv("cleaned_dataset.csv")

@app.route("/")
def home():
    # fetch the assistant’s name so index.html can show “Chatbot – <name>”
    assistant_name = fetch_assistant_name(ASSISTANT_ID)
    return render_template(
        "index.html",
        assistant_name=assistant_name,
        prediction_text=""            # no prediction yet
    )

@app.route("/predict", methods=["POST"])
def predict():
    floats   = [float(x) for x in request.form.values()]
    features = [np.array(floats)]
    pred     = model.predict(features)[0]
    msg      = f"Predicted Solar Energy: {pred:.2f} kWh"

    assistant_name = fetch_assistant_name(ASSISTANT_ID)
    return render_template(
        "index.html",
        assistant_name=assistant_name,
        prediction_text=msg
    )

@app.route("/dashboard")
def dashboard():
    county_avg = (
        data.groupby("county")["Solar_Energy2022"]
            .mean()
            .reset_index()
            .sort_values("Solar_Energy2022", ascending=False)
    )
    features = [
        "ghi_mean2022", "dni_mean2022", "dhi_mean2022",
        "temp_mean2022", "wind_mean2022", "sp_mean2022"
    ]
    feature_corr = (
        data[features + ["Solar_Energy2022"]]
            .corr()["Solar_Energy2022"]
            .drop("Solar_Energy2022")
    )
    map_data = data[["latitude", "longitude", "Solar_Energy2022"]].dropna()

    return render_template(
        "dashboard.html",
        county_avg   = county_avg.to_dict(orient="records"),
        feature_data = feature_corr.to_dict(),
        map_data     = map_data.to_dict(orient="records")
    )

# ─── Chatbot Integration ────────────────────────────────────────────────────────

# 1) Initialize the OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = openai.OpenAI(api_key=openai.api_key)
ASSISTANT_ID = "asst_8mfKWKTcUehHNUcNHGzCt1Ma"

def fetch_assistant_name(assistant_id):
    """Retrieve the assistant’s human-readable name."""
    assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
    return assistant.name

def ensure_thread_id():
    """Create (or reuse) a thread_id in Flask’s session."""
    if "thread_id" not in session:
        thread = client.beta.threads.create()
        session["thread_id"] = thread.id
    return session["thread_id"]

def stream_generator(prompt, thread_id):
    """
    Send the user prompt, then yield each chunk of the assistant’s reply.
    """
    # send the user turn
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )
    # stream assistant turn
    stream = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        stream=True
    )
    for event in stream:
        if event.data.object == "thread.message.delta":
            for chunk in event.data.delta.content:
                if chunk.type == "text":
                    yield chunk.text.value

@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives { prompt } → uses stream_generator → returns full response.
    """
    data   = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"response": ""})

    thread_id = ensure_thread_id()
    # accumulate the streamed chunks into one string
    response_text = "".join(stream_generator(prompt, thread_id))
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)

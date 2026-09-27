import os
from flask import Flask, send_from_directory

# Initialize Flask app pointing to the outputs directory where vhms_dashboard.html lives
app = Flask(__name__, static_folder="outputs")


# Route for main dashboard URL
@app.route("/")
def index():
  return send_from_directory("outputs", "vhms_dashboard.html")


# Route to serve supporting files like fleet_predictions.json
@app.route("/<path:filename>")
def serve_static(filename):
  return send_from_directory("outputs", filename)


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  app.run(host="0.0.0.0", port=port)
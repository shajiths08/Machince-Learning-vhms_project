import os
from flask import Flask, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VHMS - Vehicle Health Monitoring System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 40px;
                background: #f4f6f8;
                text-align: center;
            }

            .container {
                max-width: 800px;
                margin: auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            h1 {
                margin-bottom: 10px;
            }

            .status {
                padding: 20px;
                margin: 25px 0;
                border-radius: 10px;
                background: #e8f5e9;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h1>Vehicle Health Monitoring System</h1>

            <div class="status">
                <h2>VHMS Application is Running</h2>
                <p>Your Flask application has been deployed successfully.</p>
            </div>

            <p>Vehicle prediction data is available.</p>

        </div>
    </body>
    </html>
    """


@app.route("/fleet_predictions.json")
def predictions():
    return send_from_directory(
        BASE_DIR,
        "fleet_predictions.json"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

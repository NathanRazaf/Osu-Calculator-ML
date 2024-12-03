from flask import Flask, Response, request
from data_scraper import get_top_50_country_usernames
import requests
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

called_api = "https://calc-osu-plays.onrender.com/fetch/user/scores"

@app.route("/")
def hello_world():
    return "Osu! Scraper API is running"

@app.route("/top50", methods=["GET"])
def get_top_50():
    country = request.args.get("country")
    if not country:
        return Response("Please provide a country code", status=400)

    def generate():
        usernames = get_top_50_country_usernames(country)
        for username in usernames:
            yield f"data: Starting processing for {username}\n\n"
            print(f"Calling API for {username}")
            try:
                response = requests.get(f"{called_api}/{username}/100", stream=True, timeout=500)
                response.raise_for_status()
                
                # Process the streaming response
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            data = line.lstrip("data: ").strip()
                            parsed_data = json.loads(data)
                            yield f"data: {json.dumps(parsed_data)}\n\n"
                        except json.JSONDecodeError:
                            print(f"Failed to parse line: {line}")
                            yield f"data: {json.dumps({'error': 'Failed to parse line'})}\n\n"
            except requests.exceptions.RequestException as e:
                print(f"Error calling API for {username}: {e}")
                yield f"data: {json.dumps({'username': username, 'error': str(e)})}\n\n"
            yield f"data: Finished processing for {username}\n\n"

        yield "data: All usernames processed\n\n"

    return Response(generate(), content_type="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
from data_scraper import get_top_50_country_usernames
import requests
import json

app = Flask(__name__)

called_api = "https://calc-osu-plays.onrender.com/fetch/user/scores"

@app.route("/")
def hello_world():
    return "Osu! Scraper API is running"

@app.route("/top50", methods=["GET"]) 
def get_top_50():
    country = request.args.get("country")
    if country is None:
        return "Please provide a country code", 400
    usernames = get_top_50_country_usernames(country)
    results = []
    for username in usernames:
        result = call_api(username)
        results.append(result)
    
    return jsonify(results)

def call_api(username):
    response = requests.get(f'{called_api}/{username}/100')
    # Process the streaming response
    results = []
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                # Parse each line of the SSE stream
                data = line.lstrip("data: ").strip()
                parsed_data = json.loads(data)
                results.append(parsed_data)
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line}")

    return results


if __name__ == "__main__":
    app.run(debug=True, port=5000)
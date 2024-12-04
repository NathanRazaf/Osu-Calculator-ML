from flask import Flask, Response, request
from data_scraper import get_top_50_country_usernames, get_beatmapset_ids
import requests
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

called_api = "https://calc-osu-plays.onrender.com/fetch/user/scores"
called_api2 = "https://calc-osu-plays.onrender.com/fetch/beatmap/scores"

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
            # Stream that processing has started for this username
            yield f"data: {{'username': '{username}', 'status': 'processing'}}\n\n"
            print(f"Calling API for {username}")
            
            try:
                # Call the external API for the user's scores
                response = requests.get(f"{called_api}/{username}/100", stream=True, timeout=5000)
                response.raise_for_status()

                # Accumulate the final results for this user
                results = []

                # Process the response line by line
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            # Parse the line and check for the "Finished processing" message
                            data = line.lstrip("data: ").strip()
                            parsed_data = json.loads(data)

                            # Collect final results if it is the final message
                            if "results" in parsed_data:
                                results = parsed_data["results"]
                                break
                        except json.JSONDecodeError:
                            print(f"Failed to parse line for {username}: {line}")
                            yield f"data: {json.dumps({'username': username, 'error': 'Failed to parse line'})}\n\n"
            except requests.exceptions.RequestException as e:
                print(f"Error calling API for {username}: {e}")
                yield f"data: {json.dumps({'username': username, 'error': str(e)})}\n\n"

            # Stream the final results for this username
            yield f"data: {json.dumps({'username': username, 'results': results})}\n\n"
            print(f"Finished processing for {username}")

        # Notify the client that all usernames have been processed
        yield "data: All usernames processed\n\n"
        print("Finished processing all usernames.")

    return Response(generate(), content_type="text/event-stream")


@app.route("/beatmapset", methods=["GET"])
def get_beatmapset_scores():
    id = request.args.get("id")
    if not id:
        return Response("Please provide a beatmapset ID", status=400)
    
    def generate():
        beatmap_ids = get_beatmapset_ids(id)
        for beatmap_id in beatmap_ids:
            yield f"data: Starting processing for beatmap {beatmap_id}\n\n"
            print(f"Calling API for beatmap {beatmap_id}")
            try:
                response = requests.get(f"{called_api2}/{beatmap_id}/100", stream=True, timeout=5000)
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
                print(f"Error calling API for beatmap {beatmap_id}: {e}")
                yield f"data: {json.dumps({'beatmap_id': beatmap_id, 'error': str(e)})}\n\n"
            yield f"data: Finished processing for beatmap {beatmap_id}\n\n"

        yield "data: All beatmaps processed\n\n"

    return Response(generate(), content_type="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

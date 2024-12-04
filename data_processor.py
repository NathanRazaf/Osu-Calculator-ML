from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta

load_dotenv()

rankings_api = "https://osu.ppy.sh/api/v2/rankings/osu/performance"
cached_token = None
token_expiration = None

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["osu-pp-benchmark-data"]

projection = {
    "actualPP": 1, 
    "accPercent": 1, 
    "combo": 1, 
    "nmiss": 1, 
    "hitJudgement": 1,
    "approachRate": 1, 
    "circleSize": 1, 
    "drainRate": 1, 
    "rating": 1,
    "EZ": 1, 
    "HT": 1, 
    "HD": 1, 
    "DT": 1, 
    "NC": 1, 
    "HR": 1, 
    "FL": 1,
    "_id": 0
}

batch_size = 5000

bool_fields = ["EZ", "HT", "HD", "DT", "NC", "HR", "FL"]


# Make an API call to Osu! to get my OAuth token
def get_token():
    current_time = datetime.now()

    if cached_token is not None and token_expiration is not None and current_time < token_expiration:
        return cached_token
    
    response = requests.post(
        "https://osu.ppy.sh/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": os.getenv("OSU_CLIENT_ID"),
            "client_secret": os.getenv("OSU_CLIENT_SECRET"),
            "scope": "public"
        }
    )
    response.raise_for_status()
    # Add 86400 seconds to the current time to get the expiration time
    token_expiration = current_time + timedelta(86400)
    return response.json()["access_token"]

def get_data():
    # Get cursor for the collection
    collection = db["training-data"]
    cursor = collection.find({}, projection, batch_size=batch_size)

    batch = []  # Temporary storage for one batch
    for document in cursor:
        batch.append(document)  # Add document to the batch

        # Once the batch is full, process it
        if len(batch) == batch_size:
            batch_df = pd.DataFrame(batch)

            # Convert boolean fields to integers
            for field in bool_fields:
                if field in batch_df.columns:
                    batch_df[field] = batch_df[field].astype(int)

            yield batch_df  # Yield the processed batch
            batch = []  # Reset the batch

    # Handle remaining documents (final batch)
    if batch:
        batch_df = pd.DataFrame(batch)

        for field in bool_fields:
            if field in batch_df.columns:
                batch_df[field] = batch_df[field].astype(int)

        yield batch_df


def get_user_scores(username):
    """
    Fetches the best scores of a user and converts them into DataFrame-like rows.

    Args:
        username (str): The Osu! username.

    Returns:
        list: A list of dictionaries, where each dictionary represents a score row.
    """
    token = get_token()
    
    # Fetch user data and scores in one step
    user_data_response = requests.get(
        f"https://osu.ppy.sh/api/v2/users/{username}/osu",
        headers={"Authorization": f"Bearer {token}"}
    )
    user_data_response.raise_for_status()
    user_data = user_data_response.json()

    user_id = user_data["id"]
    scores_response = requests.get(
        f"https://osu.ppy.sh/api/v2/users/{user_id}/scores/best",
        headers={"Authorization": f"Bearer {token}"},
        params={"mode": "osu", "limit": 100}
    )
    scores_response.raise_for_status()
    scores = scores_response.json()

    # Convert scores into rows
    score_rows = [convert_score_into_df_row(score) for score in scores]

    # Convert the list of dictionaries into a DataFrame
    return pd.DataFrame(score_rows)


def convert_score_into_df_row(score):
    """
    Converts a single score dictionary from the API response into a structured format.

    Args:
        score (dict): The score data from the API response.

    Returns:
        dict: A dictionary representing the score in a structured format.
    """
    try:
        return {
            "actualPP": score.get("pp", 0),
            "accPercent": score.get("accuracy", 0) * 100,
            "combo": score.get("max_combo", 0),
            "nmiss": score["statistics"].get("count_miss", 0),
            "hitJudgement": score["beatmap"].get("accuracy", 0),
            "approachRate": score["beatmap"].get("ar", 0),
            "circleSize": score["beatmap"].get("cs", 0),
            "drainRate": score["beatmap"].get("drain", 0),
            "rating": score["beatmap"].get("difficulty_rating", 0),
            "EZ": 1 if "EZ" in score.get("mods", []) else 0,
            "HT": 1 if "HT" in score.get("mods", []) else 0,
            "HD": 1 if "HD" in score.get("mods", []) else 0,
            "DT": 1 if "DT" in score.get("mods", []) else 0,
            "NC": 1 if "NC" in score.get("mods", []) else 0,
            "HR": 1 if "HR" in score.get("mods", []) else 0,
            "FL": 1 if "FL" in score.get("mods", []) else 0
        }
    except KeyError as e:
        print(f"Missing field in score data: {e}")
        return {}
    




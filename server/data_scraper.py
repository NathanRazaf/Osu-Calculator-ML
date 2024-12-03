import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
load_dotenv()

rankings_api = "https://osu.ppy.sh/api/v2/rankings/osu/performance"
cached_token = None
token_expiration = None

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


def get_top_50_country_usernames(country):
    token = get_token()
    response = requests.get(
        rankings_api,
        headers={"Authorization": f"Bearer {token}"},
        params={ "country": country, "mode": "osu", "limit": 4 }
    )
    response.raise_for_status()
    users = response.json()["ranking"]
    # Extract the usernames from the response
    return [user["user"]["username"] for user in users]









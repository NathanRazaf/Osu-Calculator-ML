from datetime import datetime, timedelta
import requests
import time

rankings_api = "https://osu.ppy.sh/api/v2/rankings/osu/performance"
scraper_api = "https://osu-data-scraper.onrender.com/beatmapset"

cached_token = None
token_expiration = None

OSU_CLIENT_ID=36013
OSU_CLIENT_SECRET="iSGjCgfBcDV9xfagJl7wneEKHZ1DV5bff8CZb7HF"

def get_token():
    current_time = datetime.now()

    if cached_token is not None and token_expiration is not None and current_time < token_expiration:
        return cached_token
    
    response = requests.post(
        "https://osu.ppy.sh/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": OSU_CLIENT_ID,
            "client_secret": OSU_CLIENT_SECRET,
            "scope": "public"
        }
    )
    response.raise_for_status()
    # Add 86400 seconds to the current time to get the expiration time
    token_expiration = current_time + timedelta(86400)
    return response.json()["access_token"]


def safe_request(url, retries=5, delay=5, timeout=3600):
    """
    Make a safe HTTP GET request with retries.
    Args:
        url (str): The URL to make the request to.
        retries (int): Number of retries before giving up.
        delay (int): Seconds to wait between retries.
        timeout (int): Timeout for the request.
    Returns:
        requests.Response: The response object if successful.
    Raises:
        requests.exceptions.RequestException: If all retries fail.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise an error for bad responses
            return response
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"ChunkedEncodingError on attempt {attempt + 1}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed on attempt {attempt + 1}: {e}")
        
        if attempt < retries - 1:  # Don't sleep on the last attempt
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    raise requests.exceptions.RequestException(f"All {retries} retries failed for URL: {url}")


def ultimate_get_all(country, isGigaBundle=False):
    bs_ids = get_all_beatmapsets_ids(country, isGigaBundle)
    for id in bs_ids:
        # Sleep for 5 seconds to avoid rate limiting
        time.sleep(5)
        try:
            response = safe_request(f"{scraper_api}?id={id}")
            print(f"Processed beatmapset {id}")
        except requests.exceptions.RequestException as e:
            print(f"Failed for beatmapset {id} after retries: {e}")


def get_top_50_country_ids(country):
    token = get_token()
    response = requests.get(
        rankings_api,
        headers={"Authorization": f"Bearer {token}"},
        params={ "country": country, "mode": "osu", "limit": 4 }
    )
    response.raise_for_status()
    users = response.json()["ranking"]
    # Extract the ids from the response
    return [user["user"]["id"] for user in users]

def get_beatmapset_ids_player(id):
    response = requests.get(
        f"https://osu.ppy.sh/api/v2/users/{id}/scores/best",
        headers={"Authorization": f"Bearer {get_token()}"},
        params={"mode": "osu", "limit": 100}
    )
    response.raise_for_status()
    scores = response.json()
    # Extract the beatmapset ids from the response
    return [score["beatmap"]["beatmapset_id"] for score in scores]

def get_all_beatmapsets_ids(country, isGigaBundle=False):
    ids = get_top_50_country_ids(country)
    # If not isGigaBundle, only retain the first 10 ids
    ids = ids[:10] if not isGigaBundle else ids
    beatmapset_ids = []
    for id in ids:
        beatmapset_ids += get_beatmapset_ids_player(id)
    # Remove duplicates
    return list(set(beatmapset_ids))

ultimate_get_all("PL", isGigaBundle=False)
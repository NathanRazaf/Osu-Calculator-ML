from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["osu-pp-benchmark-data"]
collection = db["training-data"]

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

batch_size = 100

bool_fields = ["EZ", "HT", "HD", "DT", "NC", "HR", "FL"]

def get_data():
    # Get cursor for the collection
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





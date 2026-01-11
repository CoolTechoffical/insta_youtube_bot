import os
import time
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGODB_URI")

client = MongoClient(MONGO_URI)
db = client["highlightbot"]
jobs = db["jobs"]

def create_job(data):
    jobs.insert_one(data)

def update_job(job_id, updates):
    jobs.update_one(
        {"job_id": job_id},
        {"$set": updates}
    )

def get_active_jobs():
    return list(
        jobs.find(
            {"status": {"$in": ["processing", "waiting"]}}
        )
    )

def get_job(job_id):
    return jobs.find_one({"job_id": job_id})

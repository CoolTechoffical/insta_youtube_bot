from pymongo import MongoClient
from config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client["highlight_bot"]
jobs = db["jobs"]

def create_job(job):
    jobs.insert_one(job)

def update_job(job_id, data):
    jobs.update_one({"job_id": job_id}, {"$set": data})

def get_pending_jobs():
    return list(jobs.find({
        "status": {"$in": ["processing", "waiting"]}
    }))

def get_job(job_id):
    return jobs.find_one({"job_id": job_id})

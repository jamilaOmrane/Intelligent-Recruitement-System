import pymongo
from pymongo import MongoClient

def connect_and_insert(desired_profile, scores ) :
    client = MongoClient('mongodb://localhost:27017', 27017)
    # Get the sampleDB database
    my_db = client["PFE"]
    my_collection = my_db['Researches']
    x = my_collection.insert_one(
        {"desired_profile" : desired_profile,
        "search_result" : scores
        })
    print(x)

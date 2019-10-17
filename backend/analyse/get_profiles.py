import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

def get_full_profile(id):
    client = MongoClient('mongodb://localhost:27017', 27017)   
    my_db = client["PFE"]
    my_collection = my_db['profiles']
    profile = my_collection.find_one({"_id": ObjectId(id)})
    for i in range(len(profile["positions"])):
            p = profile["positions"][i] 
            try: 
                p["start_date"] = p["start-date"]
            except:
                pass
            try:
                p["end_date"] = p["end-date"]
            except:
                p["is_current"] = p["is-current"]
            profile["positions"][i] = p
    for i in range(len(profile["educations"])):
            p = profile["educations"][i] 
            try: 
                p["start_date"] = p["start-date"]
            except:
                pass
            try:
                p["end_date"] = p["end-date"]
            except:
                pass
            try:
                p["field_of_study"] = p["field-of-study"]
            except:
                pass
            profile["educations"][i] = p
    return profile

"""def main():
    x = get_full_profile("5b041830006b1a6d941ba205")
    print(x)
if __name__== "__main__":
  main()"""
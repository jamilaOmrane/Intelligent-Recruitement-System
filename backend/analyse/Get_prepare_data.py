import pymongo
from pymongo import MongoClient
import nltk
import re
from nltk.stem.snowball import SnowballStemmer


def connect_mongo() :
    my_profiles = []
    client = MongoClient('mongodb://localhost:27017', 27017)
    # Get the sampleDB database
    my_db = client["PFE"]
    my_collection = my_db['profiles']
    #2094 = 1/10 des profiles
    for record in my_collection.find().limit(my_collection.count()//10):
        my_profiles.append(record)
    #print(len(my_profiles))
    return my_profiles
    

def get_valid(my_profiles):
    profiles = []
    jobs = []
    job_titles = []
    for profile in my_profiles:
        positions = profile["positions"]
        for position in positions:
            if 'title' in position and 'summary' in position :
                jobs.append(position['title']+" "+position['summary'])
                job_titles.append(position['title'])
                profiles.append(profile.get('_id'))
    return profiles, jobs, job_titles

def get_perso_stopwords():
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append("'d")
    stopwords.append("'m")
    stopwords.append("'ve")
    stopwords.append("'a")
    stopwords.append("'s")
    punctuations = '''!()”“’-[];:'"\,<>./?@#$%^&*_•~+*'''
    numbers = '0123456789'
    for p in punctuations:
        stopwords.append(p)
    for n in numbers:
        stopwords.append(n)
    perso_stop_words = ['Senior', 'Engineer', 'Chief', 'Manager', 'Technician', 'Intern', 'Researcher', 'Director', 'Internship', 'Stud.', 'Temporary', 'Freelance', 'Architect', 'Trainee', 'SPECIALIST', 'Designer', 'Semi-Senior', 'Junior', 'Leader', 'Design', 'Systems', 'Bachelor', 'Student', 'Sr.', 'Head', 'Partner', 'Consultant', 'Undergraduate', 'Specialist', 'Free', 'lancer', 'Eng', 'seinor', 'joiner', 'Assisstant', 'Co-Founder', 'Master', 'Summer', 'Training', 'Graduate', 'Project', 'Jr.', 'CEO', 'VP', 'Jr', 'Sr', 'Founder', 'Fulltime', 'President', 'Team', 'Lead', 'PROJECTS', 'Part-time', 'Executive', 'Instructor', 'Supervisor', 'Advisor', 'Analyst', 'Member', 'Volunteer', 'mission', 'Services', 'Service', 'company', 'Worker', 'Staff', 'Research', 'researcher', 'PhD', 'Scholar', 'Candidate']
    for w in perso_stop_words:
        stopwords.append(w.lower())
    return stopwords

def tokenize_and_stem(text):
    stemmer = nltk.stem.SnowballStemmer('english')
    punctuations = '''!()”“’-[];:'"\,<>./?@#$%^&*_•~+*'''
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtred_tokens = []
    for token in tokens: 
        """        if "**" in token:
                    token = token = token.replace('**','')
                if "--" in token:
                    token = token = token.replace('--','')"""
        if re.search('[a-zA-Z]',token) and len(token)>=4:
            for c in punctuations:
                for t in token :
                    if(c == t):
                        token = token.replace(c,'')
            filtred_tokens.append(token)
    stems = [stemmer.stem(f) for f in filtred_tokens]
    return stems
    
    

def main():
     p = connect_mongo()
     profiles, jobs, job_titles = get_valid(p)
     print(jobs[0])

if __name__== "__main__":
  main()
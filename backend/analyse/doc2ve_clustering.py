import os
import json
import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import analyse.Get_prepare_data as gpd
from sklearn.externals import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import time



profiles = []
jobs = []
job_titles = []


def tokenize_and_filter(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtred_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]',token):
            filtred_tokens.append(token)
    return filtred_tokens

def doc2vec(jobs):
    tokenized_jobs = []
    for j in jobs:
        tokenized_jobs.append(tokenize_and_filter(j))
    model = gensim.models.Word2Vec(tokenized_jobs, min_count= 1)
    tagged_jobs = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(jobs)]
    max_epochs = 100
    vec_size = 20
    alpha = 0.025
    d2v_model = Doc2Vec(size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.025,
                    min_count=1,
                    dm =1)
    d2v_model.build_vocab(tagged_jobs)
    for epoch in range(10):
        d2v_model.train(tagged_jobs,total_examples=d2v_model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    d2v_result = d2v_model.docvecs.most_similar('1')
    print(d2v_result)    

    return d2v_model

def kmeans(num_clusters, d2v_model):
    num_clusters = num_clusters
    km = KMeans(n_clusters=num_clusters)
    start = time.time()
    km.fit(d2v_model.docvecs.doctag_syn0)
    end = time.time()
    print("Time: " + str(end - start))
    clusters = km.labels_.tolist()
    #joblib.dump(km,"kmeans_doc2vec_7718.pkl")
    return km

def new_job_prediction(new_job, d2v_model,km):
    new_job_text = [new_job["title"]+" "+new_job["summary"]]
    prediction = km.predict(d2v_model.infer_vector(new_job_text).reshape(1, -1))
    print(prediction)
    return(prediction)

def get_cluster_profiles(km, prediction, profiles, job_titles):
    clusters = km.labels_.tolist()
    jobs_frame = {'job_title': job_titles, 'profiles': profiles, 'cluster': clusters}
    frame = pd.DataFrame(jobs_frame, index = [clusters], columns=['job_title','profiles','cluster'])
    cluster_profiles = []
    for j in range(len(frame)):
            if frame.iloc[j][2] == prediction :
                #print(frame.iloc[j][0])
                if(frame.iloc[j][1] not in cluster_profiles):
                    cluster_profiles.append(frame.iloc[j][1])
    return(cluster_profiles)
    """for t in cluster_profiles:
        print(t)"""

def evaluation(labels,data):
    silhouette = metrics.silhouette_score(data, labels, sample_size=None)
    print(silhouette)
        

def main_process(desired_job):
    profile_records = gpd.connect_mongo()
    profiles, jobs, job_titles = gpd.get_valid(profile_records)
    my_doc2vec =doc2vec(jobs)
    km = joblib.load("./analyse/kmeans_doc2vec_7718.pkl")
    
    """ evaluation(km.labels_,my_doc2vec.docvecs.doctag_syn0)
   """
    
    #km = kmeans(4, my_doc2vec)
    prediction = new_job_prediction(desired_job,my_doc2vec,km)
    predicted_cluster = prediction.tolist()[0]
    cluster_profiles = get_cluster_profiles(km, predicted_cluster,profiles,job_titles)
    return cluster_profiles


"""if __name__== "__main__":
  main()"""

    
    
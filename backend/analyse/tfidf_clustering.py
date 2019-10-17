
# coding: utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
import Get_prepare_data as gpd
#from Get_prepare_data import connect_mongo, get_valid, get_perso_stopwords, tokenize_and_stem
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd
import time


profiles = []
jobs = []
job_titles = []


def tfidf(jobs):

    stopwords = gpd.get_perso_stopwords()
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=50000,
                                    min_df=0.0, stop_words=stopwords,
                                    use_idf=True, tokenizer=gpd.tokenize_and_stem, ngram_range=(1,3))

    tfidf_vectorizer.fit(jobs)
    tfidf_matrix = tfidf_vectorizer.transform(jobs)
    #print(tfidf_matrix.shape)
    terms = tfidf_vectorizer.get_feature_names()
    return tfidf_vectorizer, tfidf_matrix
    


def dissimilarity_count(tfidf_matrix):
    dist = 1 - cosine_similarity(tfidf_matrix)
    return dist

def elbow_mds(dist):
    MDS()
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)
    distortions = []
    K = range(1,50)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(pos)
        kmeanModel.fit(pos)
        distortions.append(sum(np.min(cdist(pos, kmeanModel.cluster_centers_, 'cosine'), axis=1)) / dist.shape[0])
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def elbow_pca(tfidf_matrix):
    X = tfidf_matrix.todense()
    reduced_data = PCA(n_components=2).fit_transform(X)
    distortions = []
    K = range(1,50)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(reduced_data)
        kmeanModel.fit(reduced_data)
        distortions.append(sum(np.min(cdist(reduced_data, kmeanModel.cluster_centers_, 'cosine'), axis=1)) / X.shape[0]) 
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return reduced_data

def kmeans_clustering(tfidf_matrix, num_clusters):
    #num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    start = time.time()
    km.fit(tfidf_matrix)
    end = time.time()
    print("Time: " + str(end - start))
    clusters = km.labels_.tolist()
    #joblib.dump(km,  'kmeans.pkl')
    return km, clusters 

def  print_job_clusters(jobs, job_titles, clusters, num_clusters ):
    jobs = {'job_title': job_titles, 'job': jobs, 'cluster': clusters}
    frame =pd.DataFrame(jobs, index = [clusters], columns=['job_title','job','cluster'])
    frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)
    for i in range(num_clusters):
        print('-----------------------------'+str(i)+'----------------------------')
        cluster_titles = []
        for j in range(len(frame)):
            if frame.iloc[j][2] == i :
                #print(frame.iloc[j][0])
                if(frame.iloc[j][0] not in cluster_titles):
                    cluster_titles.append(frame.iloc[j][0])
        for t in cluster_titles:
            print(t)

def visualisation_pca(tfidf_matrix, km, reduced_data):
    cluster_colors = {0: '#000000', 1: '#6E6E6E', 2: '#610B21', 3: '#DF013A', 4: '#FA58AC', 5:"#4B088A" , 6:"#3A01DF" , 7:"#2E64FE" , 8:"#00BFFF", 9:"#088A85",
                 10: '#01DF3A', 11: '#0B610B', 12: '#688A08', 13: '#D7DF01', 14: '#61380B', 15:"#DBA901" , 16:"#FF8000" , 17:"#DF0101" , 18:"#070B19", 19:"#6A0888",
                 20: '#5E610B', 21: '#d95f02', 22: '#7570b3', 23: '#e7298a', 24: '#66a61e', 25:"#08088A" , 26:"#B40404" , 27:"#0B3B17" , 28:"#151515", 29:"#210B61"}
    cluster_colors_10 = {0: '#DF0101', 1: '#FF8000', 2: '#298A08', 3: '#0489B1', 4: '#0404B4', 5:"#6A0888" , 6:"#424242" , 7:"#1C1C1C"}
    labels = km.fit_predict(tfidf_matrix)
    fig, ax = plt.subplots(figsize=(10,10))
    for index, instance in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = cluster_colors_10[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
    plt.savefig('5clusters.png', dpi=200)
    plt.show()

def visualisation_mds(tfidf_matrix, km, pos):
    cluster_colors_10 = {0: '#DF0101', 1: '#FF8000', 2: '#298A08', 3: '#0489B1', 4: '#0404B4', 5:"#6A0888" , 6:"#424242" , 7:"#1C1C1C"}
    labels = km.fit_predict(tfidf_matrix)
    fig, ax = plt.subplots(figsize=(10,10))
    for index, instance in enumerate(pos):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = pos[index]
        color = cluster_colors_10[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
    plt.savefig('4clustersMDS.png', dpi=200)
    plt.show()

def prediction_tfidf(new_job,tfidf_vectorizer,km):
    new_job_text = [new_job["title"]+" "+new_job["summary"]]
    tfidf_new_job = tfidf_vectorizer.transform(new_job_text)
    prediction = km.predict(tfidf_new_job)
    print(prediction)


def evaluation(labels,data):
    silhouette = metrics.silhouette_score(data, labels, sample_size=None)
    print(silhouette)




def main():
    profile_records = gpd.connect_mongo()
    profiles, jobs, job_titles = gpd.get_valid(profile_records)
    tfidf_vec, tfidf_matrix= tfidf(jobs)
    #dist = joblib.load("dist_7718.pkl")
    km, clusters = kmeans_clustering(tfidf_matrix,5)
    #km_loaded = joblib.load("analyse/my_kmeans.pkl")
    evaluation(km.labels_,tfidf_matrix)
    
    """new_job = {"title":"Sr. Java Developper" ,"summary":"We are looking for a Java Developer with experience in building high-performing, scalable, enterprise-grade applications.You will be part of a talented software team that works on mission-critical applications. Java developer roles and responsibilities include managing Java/Java EE application development while providing expertise in the full software development lifecycle, from concept and design to testing."}
    prediction_tfidf(new_job,tfidf_vec,km_loaded)"""
    

if __name__== "__main__":
  main()

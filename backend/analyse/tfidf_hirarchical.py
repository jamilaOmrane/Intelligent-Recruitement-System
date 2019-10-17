from scipy.cluster.hierarchy import ward, dendrogram, cut_tree, fcluster, complete, single, average
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import Get_prepare_data as gpd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.cluster.hierarchy import linkage
#import Get_prepare_data as gpd
import time
from sklearn import metrics

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
    """dist = 1 - cosine_similarity(tfidf_matrix)
    joblib.dump(dist, "analyse/dist_7718.pkl")"""
    dist = joblib.load("analyse/dist_7718.pkl")
    return dist

def hierarchical_clustering(dist):

    linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
    #linkage_matrix = complete(dist)
    """fig, ax = plt.subplots(figsize=(20, 30)) # set size
    ax = dendrogram(linkage_matrix, orientation="right")
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    plt.tight_layout() 
    plt.savefig('h_clustering_7718.png', dpi=200) """
    
    """cutree = cut_tree(linkage_matrix, n_clusters=16)
    return cutree"""
    """plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        linkage_matrix,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig('tfidf_hierarchical_clustering_average.png', dpi=200)
    plt.show()"""
    start = time.time()
    cuttree = cut_tree(linkage_matrix, n_clusters = 3)
    end = time.time()
    print("Time: " + str(end - start))
    return cuttree



def  print_job_clusters(jobs, job_titles, cuttree, num_clusters ):
    clusters = []
    for x in cuttree:
        clusters.append(x[0])
    jobs = {'job_title': job_titles, 'job': jobs, 'cluster': clusters}
    frame =pd.DataFrame(jobs, index = [clusters], columns=['job_title','job','cluster'])
    frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)
    file_object  = open("tf_idf_hierarchical.txt", "a", encoding="utf-8")
    for i in range(num_clusters):
        file_object.write('-----------------------------'+str(i)+'----------------------------')
        print('-----------------------------'+str(i)+'----------------------------\n')
        cluster_titles = []
        for j in range(len(frame)):
            if frame.iloc[j][2] == i :
                #print(frame.iloc[j][0])
                if(frame.iloc[j][0] not in cluster_titles):
                    cluster_titles.append(frame.iloc[j][0])
        for t in cluster_titles:
            print(t)
            file_object.write(t)
            file_object.write("\n")
    file_object.close()
    
def evaluation(labels,data):
    silhouette = metrics.silhouette_score(data, labels, sample_size=None)
    print(silhouette)   



def main():
    profile_records = gpd.connect_mongo()
    profiles, jobs, job_titles = gpd.get_valid(profile_records)
    tfidf_vec, tfidf_matrix= tfidf(jobs)
    dist = dissimilarity_count(tfidf_matrix)
    #hierarchical_clustering(dist)
    cuttree = hierarchical_clustering(dist)
    clusters = []
    for x in cuttree:
        clusters.append(x[0])
    evaluation(clusters, tfidf_matrix)
    """print_job_clusters(jobs, job_titles, cuttree, 16)"""

if __name__== "__main__":
  main()
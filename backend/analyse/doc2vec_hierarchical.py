import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import Get_prepare_data as gpd
from scipy.cluster.hierarchy import cut_tree
import pandas as pd
from sklearn.externals import joblib
import os  
import matplotlib as mpl
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn import manifold
from sklearn import metrics
import time

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
    joblib.dump(d2v_model, "d2v_7718.pkl") 
    return d2v_model

def hierarchical_clustering(doc2vec_model):
    
    linkage_matrix = linkage(y = doc2vec_model.docvecs.doctag_syn0, method="complete", metric='cosine') 
    """plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        linkage_matrix,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig('h_clustering_single_doc2vec.png', dpi=200)
    plt.show()"""
    """fig, ax = plt.subplots(figsize=(20, 30)) # set size
    ax = dendrogram(linkage_matrix, orientation="right")
    plt.tick_params(
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    plt.tight_layout() 
    plt.savefig('doc2vec_hierarchical_clustering_average.png', dpi=200)"""
    start = time.time()
    cuttree = cut_tree(linkage_matrix, n_clusters = 80 )
    end = time.time()
    print("Time: " + str(end - start))
    return cuttree

def vis(d2v_model,cuttree):
    pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
    datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)
    #datapoint = manifold.SpectralEmbedding(n_components=2).fit_transform(d2v_model.docvecs.doctag_syn0)
    clusters = []
    for x in cuttree:
        clusters.append(x[0])
    palette = sns.color_palette(None, 20)
    plt.figure(figsize=(10,10))
    #label1 = ['#DF0101',  '#FF8000',  '#298A08',  '#0489B1',  '#0404B4', "#6A0888" , "#424242" , "#1C1C1C"]
    """label1 = [ '#000000', '#6E6E6E',  '#610B21',  '#DF013A',  '#FA58AC', "#4B088A" , "#3A01DF" , "#2E64FE" , "#00BFFF", "#088A85",
                        '#01DF3A',  '#0B610B',  '#688A08',  '#D7DF01',  '#61380B', "#DBA901" , "#FF8000" , "#DF0101" , "#070B19", "#6A0888"]"""
    color = [palette[i] for i in clusters]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
    plt.savefig('doc2vec_hierarchical_vis_average.png', dpi=200)
    plt.show()

def  print_job_clusters(jobs, job_titles, cuttree, num_clusters ):
    clusters = []
    for x in cuttree:
        clusters.append(x[0])
    jobs = {'job_title': job_titles, 'job': jobs, 'cluster': clusters}
    frame =pd.DataFrame(jobs, index = [clusters], columns=['job_title','job','cluster'])
    frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)
    file_object  = open("doc2vec_hierarchical_complete.txt", "a", encoding="utf-8")
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
    #my_doc2vec =doc2vec(jobs)
    my_doc2vec = joblib.load("analyse/d2v_7718.pkl")
    #hierarchical_clustering(my_doc2vec)
    cuttree = hierarchical_clustering(my_doc2vec)
    #print_job_clusters(jobs, job_titles, cuttree, 80)
    #vis(my_doc2vec,cuttree)
    clusters = []
    for x in cuttree:
        clusters.append(x[0])
    evaluation(clusters, my_doc2vec.docvecs.doctag_syn0)


if __name__== "__main__":
  main()
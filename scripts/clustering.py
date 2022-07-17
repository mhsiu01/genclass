import argparse
import json
import math
import os
import pickle as pk
import random
import re

import numpy as np
import scipy.stats
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from sklearn.preprocessing import normalize
from tqdm import tqdm


from utils import (INTERMEDIATE_DATA_FOLDER_PATH, cosine_similarity_embedding,
                   cosine_similarity_embeddings, evaluate_predictions,
                   most_common, pairwise_distances)
from cluster_utils import (generate_keywords, generate_class_representation, 
                    generate_doc_representations)

'''
Returns tokenized documents, class and document representations, and vocabulary. 
'''
def importData(data_dir, lm_type, layer, document_repr_type):
    # Tokenized documents
    with open(os.path.join(data_dir, f"tokenization_lm-{lm_type}-{layer}.pk"), "rb") as f:
        tokenization_info = pk.load(f)["tokenization_info"]
    # Class and document representations
    with open(os.path.join(data_dir, f"document_repr_lm-{lm_type}-{layer}-{document_repr_type}.pk"), "rb") as f:
        dictionary = pk.load(f)
        class_representations = dictionary["class_representations"]
        document_representations = dictionary["document_representations"]
        raw_document_representations = dictionary["raw_document_representations"]
    # Corpus-dependent vocab words        
    with open(os.path.join(data_dir, f"static_repr_lm-{lm_type}-{layer}.pk"), "rb") as f:
        dictionary = pk.load(f)
        vocab_words = dictionary["vocab_words"]
    return tokenization_info, class_representations, document_representations, raw_document_representations, vocab_words
 

'''  
Partitions document representations into low and high confidence groups
'''
def partitionDataset(threshold, doc_reps, known_class_reps):
    # Matrix of cosine similarities with respect to known class representations
    cosine_similarities = cosine_similarity_embeddings(doc_reps, known_class_reps) 
    document_class_assignment = np.argmax(cosine_similarities, axis=1) #Cosine similarity predictions
    low_conf_docs = []
    high_conf_docs = []
    for i,doc_rep in enumerate(doc_reps):
        doc_tuple = (doc_rep, i)
        cosine_predicted_class = document_class_assignment[i] #Prediction out of known classes
        doc_max_similarity = cosine_similarities[i][cosine_predicted_class] #Gets actual similarity
        if doc_max_similarity >= threshold:
            high_conf_docs.append(doc_tuple)
        else:
            low_conf_docs.append(doc_tuple)

    # Check partition sizes
    print(f"Confidence threshold = {threshold}")
    print(f"Number of low confidence docs = {len(low_conf_docs)}")
    print(f"Number of high confidence docs = {len(high_conf_docs)}")

    return low_conf_docs, high_conf_docs, document_class_assignment



# def replace_with_raw(low_conf_docs, raw_doc_reps):
#     raw_low_conf_docs = []
#     for doc, index in low_conf_docs:
#         raw_low_conf_docs.append(raw_doc_reps[index])
#     return raw_low_conf_docs


def main(dataset,
         pca,
         random_state,
         lm_type,
         document_repr_type,
         layer,
         num_expected,
         naming_suffix):
    
    # Data preprocessing
    data_dir = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset)
    tokenization_info, class_representations, document_representations, raw_document_representations, vocab_words = importData(data_dir, lm_type, layer, document_repr_type)
    if pca != 0:
        _pca = PCA(n_components=pca, random_state=random_state)
        document_representations        = _pca.fit_transform(document_representations)
        raw_document_representations    = _pca.transform(raw_document_representations)
        class_representations           = _pca.transform(class_representations)
        print(f"Explained variance: {sum(_pca.explained_variance_ratio_)}")         

    # Partition document representations
    low_conf_docs, high_conf_docs, document_class_assignment = partitionDataset(0.40, document_representations, class_representations)

    # Clustering low-confidence documents using KMeans++
    kmeans = KMeans(n_clusters=(num_expected-len(class_representations)), init='k-means++', random_state=random_state)
    kmeans.fit(document_representations)

    # Generate keywords from low-confidence clusters

    # Generate class representations for new class names

    # Plot confusion matrix


    '''#####################
     MAIN LOOP: Generations
    #####################'''      
    for gen in range(1,4):
        print(f"Starting generation #{gen}")
        print(f"Initial PCA for gen{gen}")
        if do_pca:
            _pca = PCA(n_components=pca, random_state=random_state)
            document_representations        = _pca.fit_transform(document_representations_no_pca)
            raw_document_representations    = _pca.transform(raw_document_representations_no_pca)
            class_representations           = _pca.transform(class_representations_no_pca)
            print(f"Explained variance: {sum(_pca.explained_variance_ratio_)}")        
        # Partitioning dataset 
        print(f"Partitioning documents gen{gen}")
        low_conf_docs, high_conf_docs, document_class_assignment = partitionDataset(0.10, document_representations, class_representations)

        # Cluster low-confidence documents 
        print(f"Clustering lowconf gen{gen}")
        low_conf_doc_reps = replace_with_raw(low_conf_docs, raw_document_representations)
        gmm = GaussianMixture(n_components=num_expected, covariance_type='tied', random_state=random_state, n_init=30, warm_start=False, verbose=0)
        gmm.fit(low_conf_doc_reps) 
        low_conf_doc_predictions = gmm.predict(low_conf_doc_reps) 

        # Generate keywords
        print(f"Generating keywords gen{gen}")
        low_conf_indices = [ doc_tuple[1] for doc_tuple in low_conf_docs ] # Grab indices with respect to all documents of low_conf_docs
        cluster_keywords = generate_keywords(tokenization_info, low_conf_doc_predictions, low_conf_indices, num_expected, vocab_words)
        for i,keywords in enumerate(cluster_keywords):
            print(f"Cluster #{i} Words :{keywords}")
        save_dict_data[f"keywords_gen{gen}"] = cluster_keywords            
        user_kept = [int(x) for x in input("Choose which clusters to keep:\n").split()]
        print(f"User_kept = {user_kept}")
        # ^ Allows for user to choose which clusters to keep and which to toss out.
        
        # Generating class representations
        print(f"Generating low-conf class reps gen{gen}")
        low_conf_class_reps = [ generate_class_representation(keywords, lm_type, layer, data_dir) for i,keywords in enumerate(cluster_keywords) if i in user_kept ]
        print(f"low_conf_class_reps length = {len(low_conf_class_reps)}")
        # if len(low_conf_class_reps) != num_expected:
        #     print("Incorrect number of generated class representations.")
        #     return
        low_conf_class_reps = np.array(low_conf_class_reps)
        # Selecting class representations
        print(f"Matching new class reps gen{gen}")
#         from scipy.optimize import linear_sum_assignment as hungarian
#         class_rep_similarity = cosine_similarity_embeddings(low_conf_class_reps, class_representations_no_pca)
#         row_ind, col_ind = hungarian(class_rep_similarity, maximize=False)
#         row_ind is list of cluster numbers to be tossed out. Remaining row indices correspond to our new class representations

        generated_class_reps = low_conf_class_reps   # [ low_conf_class_reps[i] for i in user_kept ]
        # Finalizing class representations for next generation of document representations
        final_class_representations = np.concatenate((class_representations_no_pca, generated_class_reps))
        for i in range(num_expected):
            if i in user_kept:
                print(f"Keeping cluster #{i} with keywords: {cluster_keywords[i]}")
        print(f"final_class_representations.shape = {final_class_representations.shape}") # Should be (num_expected)x(768)
        if final_class_representations.shape != (num_expected,768):
            print("final_class_representations shape is not 9x768, not necessarily a problem.")
#             return

        # Recalculate new document representations for all documents, these are class aligned with both the known and generated classes
        print(f"Generating doc reps gen{gen}")

        final_doc_representations = generate_doc_representations(final_class_representations, attention_mechanism, lm_type, layer, data_dir)
        print(f"Saving gen{gen} representations")
        save_dict_data[f"class_representations_gen{gen}"] = final_class_representations
        save_dict_data[f"doc_representations_gen{gen}"] = final_doc_representations
        
        # Initialize representations for next generation
#         class_representations = final_class_representations
        document_representations_no_pca = final_doc_representations # Overwrite unPCA-ed doc reps for next generation
        if do_pca:
            print(f"PCA on gen{gen} class/doc reps")
            _pca = PCA(n_components=pca, random_state=random_state)
            final_doc_representations = _pca.fit_transform(final_doc_representations)
            final_class_representations = _pca.transform(final_class_representations)
            print(f"Final explained variance: {sum(_pca.explained_variance_ratio_)}")         


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="NYT-Topics")
    parser.add_argument("--pca", type=int, default=64, help="number of dimensions projected to in PCA, "
                                                            "-1 means not doing PCA.")
    # language model + layer
    parser.add_argument("--lm_type", default="bbu")
    # attention mechanism + T
    parser.add_argument("--document_repr_type", default="mixture-100")
    parser.add_argument("--attention_mechanism", type=str, default="mixture")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--num_expected", type=int, default=9)

    args = parser.parse_args()
    print(vars(args))
    naming_suffix = f"pca{pca}.clus{cluster_method}.{lm_type}-{layer}.{document_repr_type}.{random_state}"
    main(args.dataset, args.pca, args.random_state, args.lm_type, args.document_repr_type, args.layer, args.num_expected, naming_suffix)

# Create lots of small but very pure atomic clusters of document representations.
# Then use GMM to cluster these atomic clusters into clusters-of-clusters, aka "blobs".
# Observe the purity of resulting blobs.

import argparse
import json
import math
import os
import pickle as pk
import random
import re
import copy

import numpy as np
import scipy.stats
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from sklearn.preprocessing import normalize
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import (INTERMEDIATE_DATA_FOLDER_PATH, DATA_FOLDER_PATH, cosine_similarity_embedding,
                   cosine_similarity_embeddings, evaluate_predictions,
                   most_common, pairwise_distances)
from cluster_utils import (generate_keywords, generate_class_representation, 
                    generate_doc_representations)

TOKENIZATION_FILE = "tokenization_lm-bbu-12.pk"
STATIC_REPS_FILE = "static_repr_lm-bbu-12.pk"
DOC_REPS_FILE = "document_repr_lm-bbu-12-mixture-100.pk"
# LABELS_FILE = ""



def pseudo_kmeans_plusplus(kmeans_init, numDocs, nuclei, classReps, num_expected):
    print(f"kmeans_init shape = {kmeans_init.shape}")
    print(f"nuclei shape = {nuclei.shape}")
    already_assigned_nuclei = []

    for centroid in range(num_expected-len(classReps)):
        # Calculate D(x), then D(x)^2
        distances = np.repeat(nuclei[:,np.newaxis,:], len(kmeans_init), axis=1)
        print(f"distances shape after repeating nuclei: {distances.shape}")
        distances = np.linalg.norm(distances - classReps, axis=2)
        print(f"distances shape after norm: {distances.shape}")
        distances = np.amin(distances, axis=1)
        print(f"distances shape after taking minimum centroid-distance per nucleus: {distances.shape}")

        distances = np.square(distances)
        print(f"distances shape after element-wise squaring: {distances.shape}")      
        normalize_sum = np.sum(distances)
        distances = distances/normalize_sum
        # distances = normalize(distances, norm='l1', axis=0)
        print(f"distances shape after normalizing: {distances.shape}")
        print(f"distances sum after normalizing: {np.sum(distances)}")
        # Keep sampling until we get a nuclei that wasn't already chosen
        while True:
            new_centroid_ID = np.random.choice(len(nuclei), 1, p=distances)
            if new_centroid_ID not in already_assigned_nuclei:
                already_assigned_nuclei.append(new_centroid_ID)
                break
        kmeans_init = np.append(kmeans_init, nuclei[new_centroid_ID])

    print(f"kmeans_init shape = {kmeans_init.shape}")
    return kmeans_init

def main(args):
    # 0. Preprocessing: importing data
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset, DOC_REPS_FILE), "rb") as f:
        d = pk.load(f)
        rawDocReps = d["raw_document_representations"]
        classReps = d["class_representations"]
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset, "dataset.pk"), "rb") as f:
        d = pk.load(f)
        knownClassNames = d["class_names"]
    with open(os.path.join(DATA_FOLDER_PATH, args.dataset, "labels.txt"), "rb") as f:
        lines = np.loadtxt(f)
        labels = np.array(lines)
        labels = labels.astype(int)

    # 1. PCA on document and class representations
    _pca = PCA(n_components=args.pca, random_state=args.random_state)
    rawDocReps = _pca.fit_transform(rawDocReps)
    classReps  = _pca.transform(classReps)
    print(f"PCA dimension {args.pca} has explained variance: {sum(_pca.explained_variance_ratio_)}")

    # 2. Loop over atom sizes
    numDocs = len(rawDocReps)
    atomSizes = [10, 50, 100, 500, 1000]#[10, 25, 50, 100, 200, 500, 1000]
    allInits = {}
    for size in atomSizes:
        # A. Cluster dataset into atoms
        numAtoms = int(numDocs / args.atom_size)        
        kmeans_atomic = KMeans(n_clusters=numAtoms, init='k-means++', random_state=args.random_state)
        kmeans_atomic.fit(rawDocReps)
        docToAtom = kmeans_atomic.predict(rawDocReps) # Map every doc-rep to its cluster/atom
        atoms = [ set() for atom in range(numAtoms) ] # For each atom, maintain set of docs in it
        for doc,atom in enumerate(docToAtom): # Partition integer indices of all docs into the atoms' sets
            atoms[atom].add(doc)

        # B. Pin class representations
        kmeans_init = copy.deepcopy(classReps)

        # C. Use pseudo-KMeans++ to initalize remaining cluster centers
        nuclei = kmeans_atomic.cluster_centers_    
        kmeans_init = pseudo_kmeans_plusplus(kmeans_init, numDocs, nuclei, classReps, args.num_expected) # [ classRep1, classRep2, ..., kmeans++1, kmeans++2, ...]

        # D. Perform KMeans after obtaining the initialization
        kmeans_docs = KMeans(n_clusters=args.num_expected, init=kmeans_init, random_state=args.random_state)
        kmeans_docs.fit(rawDocReps)
        allInits[size] = (size, kmeans_init, kmeans_docs.inertia_, kmeans_docs.predict(rawDocReps)) # Check that kmeans_init is actually different for each iteration over atomSizes


    # 3. Keep clustering/initalization with best inertia
    bestSize = atomSizes[0]
    bestInertia = allInits[bestSize][2]
    for size in allInits:
        if allInits[size][2] < bestInertia:
            bestSize = size
            bestInertia = allInits[size][2]

    # 4. Display confusion matrix
    predictions = allInits[bestSize][3]
    conf_matrix = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(range(args.num_expected)))
    disp.plot()
    plt.title(f"{args.dataset_name}, rawDocReps, pseudo-kmeans++, atomSize={bestSize}")
    plt.xticks(rotation=45)
    plt.show()    

    # Run Kmeans clustering ---------------------------[OLD]------------------------------------------------------
    # numDocs = len(rawDocReps)
    # numAtoms = int(numDocs / args.atom_size)
    # kmeans = KMeans(n_clusters=numAtoms, init='k-means++', random_state=args.random_state)
    # kmeans.fit(rawDocReps)
    # docToAtom = kmeans.predict(rawDocReps) # Map every doc-rep to its atom
    # atoms = [ set() for atom in range(numAtoms) ] # Maintain list of docs in each atom
    # for doc,atom in enumerate(docToAtom): # Partition integer indices of all docs into the atoms' sets
    #     atoms[atom].add(doc)
    # print("kmeans done.")
    # Run Kmeans clustering ---------------------------[OLD]------------------------------------------------------

# Evaluate purity and plot purity distribution of atomic clusters
    # atomSizes = [ len(atom) for atom in atoms ]
    # atomPurities = []
    # atomMajorities = []
    # for atom in atoms:
    #     assignments = [ labels[doc] for doc in atom ] # Create list of groundtruth labels of all docs for that atom
    #     c = Counter(assignments)
    #     mostCommonLabel,maxLabelCount = c.most_common(1)[0] # Class that shows up most in this atom, and how many times it showed up
    #     atomMajorities.append(mostCommonLabel) # List of most common class in each atom
    #     atomPurities.append(maxLabelCount / len(atom)) # Store tuple of purity and which class the atom was.
    # print("atom evaluation done.")
    
    # plt.hist(atomPurities)
    # plt.title("Distribution of atom purities")
    # plt.show()

''' GMM performed poorly when used to cluster atoms.
# Perform GMM on atomic clusters
    nuclei = kmeans.cluster_centers_ # Array of atom centers
    print(f"nuclei shape is {nuclei.shape}")
    num_blobs = args.num_expected
    gmm = GaussianMixture(n_components=num_blobs, 
                          covariance_type='tied',
                          random_state=args.random_state,
                          n_init=999)
    gmm.fit(nuclei)
    atomToBlob = gmm.predict(nuclei) # Map every nucleus to a blob
    blobs = [ set() for blob in range(num_blobs) ]
    # Currently not maintaining a set of atoms per blob, instead doing documents per blob.
    for atom,blob in enumerate(atomToBlob): # For each atom and the blob to which atom is assigned...
        blobs[blob] = blobs[blob].union(atoms[atom])
    print("gmm done.")
# Evaluate blob purity
    blobSizes = [ len(blob) for blob in blobs ]
    blobPurities = []
    blobMajorities = []
    for blob in blobs:
        assignments = [ labels[doc] for doc in blob ] # Create list of groundtruth labels of all docs for that blob
        c = Counter(assignments)
        mostCommonLabel,maxLabelCount = c.most_common(1)[0] # Class that shows up most in this blob, and how many times it showed up
        blobMajorities.append(mostCommonLabel) # List of most common class in each blob
        blobPurities.append(maxLabelCount / len(blob)) # Store tuple of purity and which class the blob was.
    print("blob evaluation done.")
    for i in range(len(blobs)):
        print(f"Blob #{i} had purity {blobPurities[i]} and majority class {blobMajorities[i]}")
    
    plt.hist(blobPurities)
    plt.title("Distribution of blob purities")
    plt.show()    
'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Nonfixed arguments:
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--atom_size", type=int, default=50)
    # language model + layer + attention mechanism + T
    parser.add_argument("--lm_type", default="bbu")
    parser.add_argument("--document_repr_type", default="mixture-100")
    parser.add_argument("--attention_mechanism", type=str, default="mixture")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--num_expected", type=int, default=9)    
    args = parser.parse_args()
    main(args)
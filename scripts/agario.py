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

from utils import (INTERMEDIATE_DATA_FOLDER_PATH, DATA_FOLDER_PATH, cosine_similarity_embedding,
                   cosine_similarity_embeddings, evaluate_predictions,
                   most_common, pairwise_distances)
from cluster_utils import (generate_keywords, generate_class_representation, 
                    generate_doc_representations)

TOKENIZATION_FILE = "tokenization_lm-bbu-12.pk"
STATIC_REPS_FILE = "static_repr_lm-bbu-12.pk"
DOC_REPS_FILE = "document_repr_lm-bbu-12-mixture-100.pk"


def main(args):
# Preprocessing: importing data, PCA
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset, DOC_REPS_FILE), "rb") as f:
        d = pk.load(f)
        rawDocReps = d["raw_document_representations"]
        classReps = d["class_representations"]
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset, STATIC_REPS_FILE), "rb") as f:
        d = pk.load(f)
        knownClassNames = d["class_names"]
    with open(os.path.join(DATA_FOLDER_PATH, args.dataset, "labels.txt"), "rb") as f:
        lines = np.loadtxt(f)
        labels = np.array(lines)
        labels = labels.astype(int)

# Start with document reps, raw.
    _pca = PCA(n_components=args.pca, random_state=args.random_state)
    rawDocReps = _pca.fit_transform(rawDocReps)
    classReps  = _pca.transform(classReps)

# Run Kmeans clustering
    numDocs = len(rawDocReps)
    numAtoms = int(numDocs / args.atom_size)
    kmeans = KMeans(n_clusters=numAtoms, init='k-means++', random_state=args.random_state)
    kmeans.fit(rawDocReps)
    docToAtom = kmeans.predict(rawDocReps) # Map every doc-rep to its atom
    atoms = [ set() for atom in range(numAtoms) ] # Maintain list of docs in each atom
    for doc,atom in enumerate(docToAtom): # Partition integer indices of all docs into the atoms' sets
        atoms[atom].add(doc)

# Evaluate purity and plot purity distribution of atomic clusters
    atomSizes = [ len(atom) for atom in atoms ]
    purities = []
    majorities = []
    for atom in atoms:
        assigments = [ labels[doc] for doc in atom ] # Create list of groundtruth labels of all docs for that atom
        c = Counter(assignments)
        mostCommonLabel,maxLabelCount = c.most_common(1)[0] # Class that shows up most in this atom, and how many times it showed up
        majorities.append(mostCommonLabel) # List of most common class in each atom
        purities.append((maxLabelCount / len(atom)) # Store tuple of purity and which class the atom was.
    
    plt.hist(purities)
    plt.set(title="Distribution of atom purities")
    plt.show()


# Perform GMM on atomic clusters
# Evaluate blob purity


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
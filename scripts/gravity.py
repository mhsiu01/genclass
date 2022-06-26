import argparse
import json
import math
import os
import pickle as pk
import random
import re
from collections import Counter

import numpy as np
import scipy.stats
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from sklearn.preprocessing import normalize
from tqdm import tqdm
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
        orientedDocReps = d["document_representations"]
        classReps = d["class_representations"]
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset, "dataset.pk"), "rb") as f:
        d = pk.load(f)
        knownClassNames = d["class_names"]
    with open(os.path.join(DATA_FOLDER_PATH, args.dataset, "labels.txt"), "rb") as f:
        lines = np.loadtxt(f)
        labels = np.array(lines)
        labels = labels.astype(int)

# Start with document reps, perform PCA.
    _pca = PCA(n_components=args.pca, random_state=args.random_state)
    rawDocReps      = _pca.fit_transform(rawDocReps)
    orientedDocReps = _pca.transform(orientedDocReps)
    classReps       = _pca.transform(classReps)
    print("pca done.")

# Find displacement of doc reps before-and-after class orientation.
    distChanges = [np.linalg.norm(orientedDocReps[i]-rawDocReps[i]) for i in range(len(rawDocReps))] # Euclidean distance between before-and-after of a doc rep
    rawCosSim      = cosine_similarity_embeddings(rawDocReps, classReps)      # All cosine similarities
    orientedCosSim = cosine_similarity_embeddings(orientedDocReps, classReps)
    maxRawSim      = np.amax(rawCosSim, axis=1) # Max cosine similarities
    maxOrientedSim = np.amax(orientedCosSim, axis=1)
    cosChanges = [ maxOrientedSim[i]-maxRawSim[i] for i in range(len(rawDocReps))] # Changes in max cosine similarity to a known class rep after class orientation
    
    plt.hist(distChanges)
    plt.title("Distribution of Euclidean distance displacements")
    plt.show()

    plt.hist(cosChanges)
    plt.title("Distribution of max cosine similarity displacements")
    plt.show()    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Nonfixed arguments:
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pca", type=int, default=64)
    # parser.add_argument("--atom_size", type=int, default=50)

    # language model + layer + attention mechanism + T
    parser.add_argument("--lm_type", default="bbu")
    parser.add_argument("--document_repr_type", default="mixture-100")
    parser.add_argument("--attention_mechanism", type=str, default="mixture")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--random_state", type=int, default=42)
    # parser.add_argument("--num_expected", type=int, default=9)

    args = parser.parse_args()
    main(args)
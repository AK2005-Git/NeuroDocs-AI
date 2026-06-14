import faiss
import pickle
import json
import numpy as np


FAISS_PATH = "indexes/faiss_index.bin"
METADATA_PATH = "indexes/metadata.pkl"
BM25_PATH = "indexes/bm25.pkl"


def load_faiss():

    return faiss.read_index(
        FAISS_PATH
    )


def save_faiss(index):

    faiss.write_index(
        index,
        FAISS_PATH
    )


def load_metadata():

    with open(
        METADATA_PATH,
        "rb"
    ) as f:

        return pickle.load(f)


def save_metadata(metadata):

    with open(
        METADATA_PATH,
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f
        )
"""
Phase 3 Step 4:
Build temporal link prediction features.

Feature:
- Hadamard product
- Absolute difference
- Squared difference
- Cosine similarity

Dimension:
64 + 64 + 64 + 1 = 193
"""

import csv
import pickle
from pathlib import Path
import sys

import numpy as np


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


from scripts.config import PROJECT_ROOT
from scripts.feature_utils import build_pair_feature, cosine_similarity


PAIRS_FILE = PROJECT_ROOT / "data/training_pairs/pairs_labeled.csv"
EMBEDDING_DIR = PROJECT_ROOT / "data/embeddings"

OUTPUT_FILE = PROJECT_ROOT / "data/training_pairs/features.npz"



def main():

    rows = list(
        csv.DictReader(
            open(PAIRS_FILE, encoding="utf-8")
        )
    )


    print(f"Processing {len(rows)} pairs")


    cache = {}


    def load_embedding(year):

        if year not in cache:

            path = EMBEDDING_DIR / f"emb_{year}.pkl"

            with open(path,"rb") as f:
                cache[year]=pickle.load(f)

        return cache[year]



    X=[]
    cos=[]
    y=[]
    years=[]



    for r in rows:

        year=int(r["cutoff_year"])

        emb=load_embedding(year)


        a=r["node_a"]
        b=r["node_b"]


        if a not in emb or b not in emb:
            continue


        va=emb[a]
        vb=emb[b]


        X.append(
            build_pair_feature(
                va,
                vb
            )
        )


        cos.append(
            cosine_similarity(
                va,vb
            )
        )


        y.append(
            int(r["label"])
        )


        years.append(year)



    X=np.array(X)
    cos=np.array(cos)
    y=np.array(y)
    years=np.array(years)



    np.savez(
        OUTPUT_FILE,
        X=X,
        cos_sim=cos,
        y=y,
        cutoff_year=years
    )


    print("==============================")
    print("Feature shape:",X.shape)
    print(
        "Positive:",
        y.sum()
    )
    print(
        "Negative:",
        (y==0).sum()
    )
    print("==============================")



if __name__=="__main__":
    main()
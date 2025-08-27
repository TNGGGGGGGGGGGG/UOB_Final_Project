# cluster_expand.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def select_diverse(
    df_single: pd.DataFrame,
    n_clusters,
    model_name: str = "intfloat/e5-base-v2"
):
    texts = df_single["text"].tolist()
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    rep_indices = []
    for cid in range(n_clusters):
        idxs = np.where(labels == cid)[0]
        dists = np.linalg.norm(embeddings[idxs] - centers[cid], axis=1)
        rep_indices.append(idxs[np.argmin(dists)])

    df_diverse = df_single.loc[rep_indices].reset_index(drop=True)
    return df_diverse, embeddings, rep_indices

def expand_by_knn(
    df_single,
    embeddings,
    rep_indices,
    n_neighbors
):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(embeddings)
    expanded_idxs = set()
    for idx in rep_indices:
        _, neigh = nbrs.kneighbors([embeddings[idx]], n_neighbors=n_neighbors)
        expanded_idxs.update(neigh[0])
    df_expanded = df_single.loc[list(expanded_idxs)].reset_index(drop=True)
    return df_expanded

def main(
    input_path: str = "acl_data_handling/emnlp_train_cleaned.csv",
    diversified_path: str = "acl_data_handling/diversified_single.csv",
    expanded_path: str = "acl_data_handling/expanded_dataset.csv",
    n_clusters: int = 64,
    n_neighbors: int = 4,
    embed_model: str = "intfloat/e5-base-v2"
):
    df_single = pd.read_csv(input_path)
    print(f"[+] Loaded {len(df_single)} single samples from '{input_path}'")

    df_diverse, embeddings, rep_indices = select_diverse(
        df_single, n_clusters=n_clusters, model_name=embed_model
    )
    df_diverse.to_csv(diversified_path, index=False)
    print(f"[+] Selected {len(df_diverse)} diverse samples → '{diversified_path}'")

    df_expanded = expand_by_knn(
        df_single, embeddings, rep_indices, n_neighbors=n_neighbors
    )
    df_expanded.to_csv(expanded_path, index=False)
    print(f"[+] Expanded to {len(df_expanded)} samples via KNN → '{expanded_path}'")

if __name__ == "__main__":
    main()

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Simulasi playlist dan kandidat
np.random.seed(42)
playlist = np.random.rand(3, 7)
candidates = np.random.rand(20, 7)
all_songs = np.vstack([playlist, candidates])

# Clustering semua lagu ke dalam 4 klaster
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(all_songs)

# Cek klaster dominan dari lagu dalam playlist (indeks 0-2)
playlist_clusters = cluster_labels[:len(playlist)]
dominant_cluster = np.bincount(playlist_clusters).argmax()

# Filter kandidat dari klaster dominan
filtered_candidates = []
filtered_indices = []
for i in range(len(candidates)):
    cluster_idx = cluster_labels[len(playlist) + i]
    if cluster_idx == dominant_cluster:
        filtered_candidates.append(candidates[i])
        filtered_indices.append(i)

# Hitung similarity ke playlist dari kandidat dalam klaster dominan
scores = []
for idx, cand_vec in zip(filtered_indices, filtered_candidates):
    score = np.mean(cosine_similarity([cand_vec], playlist)[0])
    scores.append((idx, score))

# Urutkan skor dari tinggi ke rendah
scores.sort(key=lambda x: x[1], reverse=True)

# Tampilkan 6 teratas
print("\n=== Rekomendasi Lagu (Cluster-based) ===")
for rank, (i, score) in enumerate(scores[:6], 1):
    if i < 9:
        print(f"{rank}. Lagu  #{i+1} | Similarity: {score:.4f} | Klaster: {cluster_labels[len(playlist)+i]+1}")
    else:
        print(f"{rank}. Lagu #{i+1} | Similarity: {score:.4f} | Klaster: {cluster_labels[len(playlist)+i]+1}")
print()
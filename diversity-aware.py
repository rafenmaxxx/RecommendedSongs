import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simulasi playlist dan kandidat
np.random.seed(42)
playlist = np.random.rand(3, 7)
candidates = np.random.rand(20, 7)
similarities = [np.mean(cosine_similarity([c], playlist)) for c in candidates]

# Ambil 10 kandidat paling mirip dan urutkan dari nilai tertinggi
top_n = 10
top_indices = np.argsort(similarities)[-top_n:][::-1]

# Nilai alpha: semakin tinggi, semakin ketat penalti untuk diversitas
alpha = 0.5
selected = []
while len(selected) < 6:
    best_candidate = None
    best_score = -np.inf

    for idx in top_indices:
        if idx in selected:
            continue

        sim_to_playlist = similarities[idx]
        sim_to_selected = 0

        if selected:
            sim_to_selected = np.mean(
                [cosine_similarity([candidates[idx]], [candidates[j]])[0][0] for j in selected]
            )

        # Penalti jika terlalu mirip dengan kandidat yang sudah dipilih sesuai nilai alpha
        score = sim_to_playlist - alpha * sim_to_selected

        if score > best_score:
            best_score = score
            best_candidate = idx

    selected.append(best_candidate)

print("\n=== Rekomendasi Lagu (Diversity-aware) ===")
for i in selected:
    if i < 9:
        print(f"Lagu  #{i+1}, Similarity to Playlist: {similarities[i]:.4f}")
    else:
        print(f"Lagu #{i+1}, Similarity to Playlist: {similarities[i]:.4f}")
print()
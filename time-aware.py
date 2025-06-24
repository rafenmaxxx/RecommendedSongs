import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

# Simulasi playlist dan kandidat
np.random.seed(42)
playlist = np.random.rand(3, 7)
candidates = np.random.rand(10, 7)

# Waktu terakhir lagu kandidat didengar (random antara 1 hingga 30 hari lalu)
now = datetime.now()
last_played = [now - timedelta(days=int(x)) for x in np.random.randint(1, 31, size=10)]

# Fungsi scoring dengan time-aware
def score_with_time(candidate, playlist, last_played_time, alpha=0.5):
    sim = np.mean(cosine_similarity([candidate], playlist))
    days_ago = (now - last_played_time).days
    time_score = 1 / (1 + days_ago)  # semakin lama, semakin kecil
    final_score = alpha * sim + (1 - alpha) * time_score
    return final_score, sim, time_score

scores = []
for i in range(len(candidates)):
    final, sim, time = score_with_time(candidates[i], playlist, last_played[i])
    scores.append((i, final, sim, time))

scores.sort(key=lambda x: x[1], reverse=True)
print("\n=== Rekomendasi Lagu (Time-aware) ===")
for rank, (i, final, sim, time) in enumerate(scores[:6], 1):
    if i < 9:
        print(f"{rank}. Lagu  #{i+1} | Skor total: {final:.4f} | Similarity: {sim:.4f} | Time Score: {time:.4f}")
    else:
        print(f"{rank}. Lagu #{i+1} | Skor total: {final:.4f} | Similarity: {sim:.4f} | Time Score: {time:.4f}")
print()
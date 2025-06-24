import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Buat 100 vector lagu dengan fitur random (rentang [0, 1])
np.random.seed(42)
song_vectors = np.round(np.random.rand(100, 7), 4)
song_titles = [f"Lagu #{i+1}" for i in range(len(song_vectors))]

initial_vectors = song_vectors[:20]
for i in range(len(song_titles)):
    if i < 9:
        song_titles[i] = song_titles[i].replace("Lagu", "Lagu ")

# Input jumlah lagu awal dan indeks
num_initial = int(input("\nMasukkan jumlah lagu yang akan dimasukkan ke playlist awal (1-20): "))
selected_indices = list(map(int, input(f"Masukkan {num_initial} indeks lagu (pisahkan dengan spasi) (1-20): ").split()))

# Ubah ke indeks 0
for i in range(len(selected_indices)):
    selected_indices[i] -= 1

# Validasi input
if len(selected_indices) != num_initial or not all(0 <= idx < len(initial_vectors) for idx in selected_indices):
    print("Input tidak valid.")
    exit()

# Inisialisasi playlist dan kandidat
playlist = np.array([initial_vectors[i] for i in selected_indices])
playlist_index = np.array([i for i in selected_indices])

mean_initial_playlist = np.round(np.mean(playlist, axis=0), 4)

while True:
    recommended_indices = []
    for step in range(6):
        best_score = -1
        best_index = -1

        for i in range(len(song_vectors)):
            if any(i == rec[0] for rec in recommended_indices) or i in playlist_index:
                continue
            avg = np.mean(cosine_similarity([song_vectors[i]], playlist)[0])
            if avg > best_score:
                best_score = avg
                best_index = i

        recommended_indices.append([best_index, best_score])

    # Isi playlist saat ini
    print("\n=== LAGU DALAM PLAYLIST ===")
    for i in range(len(playlist)):
        print(f"{i + 1}. {song_titles[playlist_index[i]]}: {playlist[i]}")

    print("\n=== 6 LAGU YANG DIREKOMENDASIKAN ===")
    for i, (idx, score) in enumerate(recommended_indices, 1):
        print(f"{i}. {song_titles[idx]}: {song_vectors[idx]}")
        print(f"   Rata-rata similarity: {score:.4f}\n")

    selected_input = input("\nMasukkan indeks (1-6) lagu yang ingin ditambahkan ke playlist (0 untuk selesai): ").strip()
    if selected_input == "0":
        print("\nProses dihentikan oleh pengguna.")
        break

    try:
        chosen = list(map(int, selected_input.split()))
        if not all(1 <= x <= 6 for x in chosen):
            raise ValueError
    except ValueError:
        print("Input tidak valid.")
        continue

    for x in chosen:
        idx, _ = recommended_indices[x-1]
        playlist = np.vstack([playlist, song_vectors[idx]])
        playlist_index = np.append(playlist_index, idx)

    print("\nPlaylist diperbarui.")
    print("Total lagu sekarang:", len(playlist))

# Ringkasan akhir
print("\n=== PLAYLIST AKHIR ===")
for i in range(len(playlist)):
    print(f"{i+1}. {song_titles[playlist_index[i]]}")
    if i == len(selected_indices) - 1 and i != len(playlist) - 1:
        print(" --- recommended songs ---")

mean_final_playlist = np.round(np.mean(playlist, axis=0), 4)
print("\nRata-rata playlist awal :", mean_initial_playlist)
print("Rata-rata playlist akhir:", mean_final_playlist)
print("Perbedaan rata-rata:", np.round(mean_final_playlist - mean_initial_playlist, 4), "\n")

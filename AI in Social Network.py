import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ===============================
# 1. DATASET LATIHAN
# ===============================

data = {
    "views":  [500000, 1500000, 800000, 3000000, 2000000, 900000, 1200000, 2500000],
    "likes":  [100000, 300000, 150000, 500000, 250000, 100000, 220000, 400000],
    "label":  [0,        1,       0,       1,       1,       0,      1,      1],  
}
# 1 = Populer, 0 = Tidak Populer

df = pd.DataFrame(data)

# ===============================
# 2. PERSIAPAN DATA
# ===============================

X = df[["views", "likes"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 3. TRAINING MODEL
# ===============================

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model Machine Learning berhasil dilatih!\n")

# ===============================
# 4. FUNGSI PREDIKSI
# ===============================

def cek_video_populer(views, likes):
    # Gunakan DataFrame agar tidak muncul warning fitur
    input_df = pd.DataFrame({"views": [views], "likes": [likes]})
    pred = model.predict(input_df)[0]
    return "Populer" if pred == 1 else "Tidak Populer"


# ===============================
# 5. INPUT PENGGUNA
# ===============================

try:
    views_input = int(input("Masukkan jumlah views: "))
    likes_input = int(input("Masukkan jumlah likes: "))

    hasil = cek_video_populer(views_input, likes_input)

    print("\n============================")
    print("HASIL PREDIKSI VIDEO")
    print("============================")
    print(f"Views  : {views_input}")
    print(f"Likes  : {likes_input}")
    print(f"Kategori Video : {hasil}")
    print("============================\n")

except ValueError:
    print("Input tidak valid. Masukkan angka saja.")
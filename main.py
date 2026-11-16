"""
PROYEK: PREDIKSI KELULUSAN MAHASISWA
Materi evaluasi: Confusion Matrix, Accuracy, Precision, Recall,
F1-Score, ROC Curve, dan AUC.

CATATAN: Dataset yang digunakan adalah data SIMULASI.
Hasil program bukan penilaian atau prediksi mahasiswa sungguhan.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ---------------------------------------------------------
# 1. MEMBACA DATASET
# ---------------------------------------------------------
DATA_PATH = Path("data/dataset_kelulusan_mahasiswa.csv")
HASIL_DIR = Path("hasil")
HASIL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset belum ditemukan. Jalankan terlebih dahulu: "
        "python generate_dataset.py"
    )

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("PREDIKSI KELULUSAN MAHASISWA")
print("=" * 65)
print("\nLima baris pertama dataset:")
print(df.head())
print(f"\nJumlah data: {len(df)}")
print("\nDistribusi kelas:")
print(df["status_kelulusan"].value_counts())

# ---------------------------------------------------------
# 2. MEMISAHKAN FITUR DAN TARGET
# ---------------------------------------------------------
fitur = [
    "kehadiran_persen",
    "nilai_tugas",
    "nilai_uts",
    "jam_belajar_per_minggu",
    "jumlah_terlambat"
]
target = "status_kelulusan"

X = df[fitur]
y = df[target]

# Label encoding: Lulus = 1, Tidak Lulus = 0
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nKode kelas:")
for kelas, kode in zip(encoder.classes_, encoder.transform(encoder.classes_)):
    print(f"{kelas} = {kode}")

# ---------------------------------------------------------
# 3. MEMBAGI DATA TRAINING DAN TESTING
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"\nData training: {len(X_train)}")
print(f"Data testing : {len(X_test)}")

# ---------------------------------------------------------
# 4. MEMBANGUN MODEL MACHINE LEARNING
# ---------------------------------------------------------
# Pipeline: mengisi data kosong, normalisasi, lalu Logistic Regression.
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. MELAKUKAN PREDIKSI
# ---------------------------------------------------------
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 6. CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX")
print(cm)

print(f"True Negative  (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive  (TP): {tp}")

# Menampilkan grafik Confusion Matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Lulus", "Tidak Lulus"]
)

display.plot(cmap="Purples", values_format="d")
plt.title("Confusion Matrix - Prediksi Kelulusan Mahasiswa")
plt.tight_layout()
plt.savefig(HASIL_DIR / "confusion_matrix.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 7. MENGHITUNG ACCURACY, PRECISION, RECALL, F1-SCORE
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 65)
print("HASIL EVALUASI MODEL")
print("=" * 65)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Lulus", "Tidak Lulus"],
    zero_division=0
))

# ---------------------------------------------------------
# 8. ROC CURVE DAN AUC
# ---------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_probability)
auc = roc_auc_score(y_test, y_probability)

print(f"ROC AUC   : {auc:.4f}")


# ---------------------------------------------------------
# 10. MENYIMPAN GRAFIK ROC CURVE
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Tebakan acak")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve - Prediksi Kelulusan Mahasiswa")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(HASIL_DIR / "roc_curve.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 11. MENYIMPAN RINGKASAN METRIK
# ---------------------------------------------------------
ringkasan = pd.DataFrame({
    "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"],
    "Nilai": [accuracy, precision, recall, f1, auc]
})
ringkasan.to_csv(HASIL_DIR / "ringkasan_evaluasi.csv", index=False)

# Simpan hasil aktual dan prediksi untuk pemeriksaan
hasil_prediksi = X_test.copy()
hasil_prediksi["aktual"] = encoder.inverse_transform(y_test)
hasil_prediksi["prediksi"] = encoder.inverse_transform(y_pred)
hasil_prediksi["probabilitas_lulus"] = y_probability
hasil_prediksi.to_csv(HASIL_DIR / "hasil_prediksi.csv", index=False)

# ---------------------------------------------------------
# 12. INTERPRETASI OTOMATIS
# ---------------------------------------------------------
print("\n" + "=" * 65)
print("INTERPRETASI HASIL")
print("=" * 65)
print(f"1. Accuracy {accuracy:.2%}: proporsi seluruh prediksi yang benar.")
print(f"2. Precision {precision:.2%}: ketepatan prediksi pada kelas Lulus.")
print(f"3. Recall {recall:.2%}: proporsi mahasiswa yang benar-benar Lulus "
      "dan berhasil dikenali model.")
print(f"4. F1-Score {f1:.2%}: keseimbangan Precision dan Recall.")
print(f"5. AUC {auc:.3f}: kemampuan model membedakan kelas pada berbagai threshold.")

print("\nPerhatian interpretasi:")
print("- False Positive: mahasiswa aktual Tidak Lulus diprediksi Lulus.")
print("- False Negative: mahasiswa aktual Lulus diprediksi Tidak Lulus.")
print("- Untuk intervensi akademik, kampus juga perlu memperhatikan kemampuan "
      "mendeteksi mahasiswa yang berisiko Tidak Lulus.")
print("- Metrik yang dicetak di atas memakai kelas positif Lulus. "
      "Jika fokus penelitian adalah mendeteksi Tidak Lulus, definisi kelas "
      "positif dan metrik perlu disesuaikan.")

print("\nFile hasil tersimpan di folder:", HASIL_DIR.resolve())
print("Dataset simulasi tidak boleh dianggap sebagai data kampus nyata.")

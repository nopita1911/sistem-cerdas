"""
Membuat dataset SIMULASI akademik mahasiswa.
Bukan data mahasiswa sungguhan.
Jalankan: python generate_dataset.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
JUMLAH_DATA = 1000

def main():
    rng = np.random.default_rng(SEED)

    kehadiran = np.clip(rng.normal(82, 12, JUMLAH_DATA), 40, 100)
    nilai_tugas = np.clip(rng.normal(75, 13, JUMLAH_DATA), 25, 100)
    nilai_uts = np.clip(rng.normal(70, 15, JUMLAH_DATA), 20, 100)
    jam_belajar = np.clip(rng.normal(10, 4, JUMLAH_DATA), 0, 30)
    jumlah_terlambat = np.clip(rng.poisson(2.2, JUMLAH_DATA), 0, 12)

    # Skor laten untuk membentuk label simulasi.
    # Nilai akhir dipengaruhi oleh fitur akademik dan variasi acak.
    noise = rng.normal(0, 9, JUMLAH_DATA)
    skor = (
        0.22 * kehadiran
        + 0.25 * nilai_tugas
        + 0.38 * nilai_uts
        + 1.15 * jam_belajar
        - 1.1 * jumlah_terlambat
        + noise
    )

    # Ambang dibuat agar kelas lulus dan tidak lulus sama-sama muncul.
    ambang = np.quantile(skor, 0.35)
    status = np.where(skor >= ambang, "Lulus", "Tidak Lulus")

    df = pd.DataFrame({
        "kehadiran_persen": np.round(kehadiran, 1),
        "nilai_tugas": np.round(nilai_tugas, 1),
        "nilai_uts": np.round(nilai_uts, 1),
        "jam_belajar_per_minggu": np.round(jam_belajar, 1),
        "jumlah_terlambat": jumlah_terlambat.astype(int),
        "status_kelulusan": status
    })

    output = Path("data") / "dataset_kelulusan_mahasiswa.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Dataset berhasil dibuat: {output}")
    print(f"Jumlah data: {len(df)} mahasiswa")
    print("\nDistribusi status:")
    print(df["status_kelulusan"].value_counts())
    print("\nContoh data:")
    print(df.head())

if __name__ == "__main__":
    main()

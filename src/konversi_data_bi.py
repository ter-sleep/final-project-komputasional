"""Konversi file unduhan Bank Indonesia -> data/inflasi.csv siap pakai."""
import pandas as pd

# Ganti nama file sesuai hasil unduhan Anda
df = pd.read_excel('Data Inflasi.xlsx')  # atau pd.read_csv jika CSV
print(df.head())  # cek dulu nama kolomnya, sesuaikan baris di bawah

df.columns = ['tanggal', 'inflasi']

# Bersihkan kolom inflasi: "3,34 %" -> 3.34
df['inflasi'] = (df['inflasi'].astype(str)
                 .str.replace('%', '', regex=False)
                 .str.replace(',', '.', regex=False)
                 .str.strip()
                 .astype(float))

# Konversi "Juni 2026" -> 2026-06 (mapping bulan Indonesia)
bulan_map = {'Januari': '01', 'Februari': '02', 'Maret': '03', 'April': '04',
             'Mei': '05', 'Juni': '06', 'Juli': '07', 'Agustus': '08',
             'September': '09', 'Oktober': '10', 'November': '11', 'Desember': '12'}


def parse_periode(t):
    nama, tahun = t.strip().split()
    return f"{tahun}-{bulan_map[nama]}"


df['periode'] = df['tanggal'].apply(parse_periode)
df = df[['periode', 'inflasi']].sort_values('periode').reset_index(drop=True)

df.to_csv('../data/inflasi.csv', index=False)
print(df.head(), f"\nTotal data: {len(df)} bulan")

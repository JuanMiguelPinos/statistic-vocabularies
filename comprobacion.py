from pathlib import Path

root = Path.cwd()

tables = list((root / "data/raw/tables_small").glob("*.csv"))
titles = list((root / "data/raw/titles").glob("*"))
nuts = list((root / "data/external/nuts").glob("*"))

print("Tablas:", len(tables))
print("Títulos:", titles)
print("NUTS:", nuts)
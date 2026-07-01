import pandas as pd

# Leer el dataset original
df = pd.read_csv("results.csv")

# Convertir la columna de fecha
df["date"] = pd.to_datetime(df["date"])

# Conservar solo partidos desde 2023
df = df[df["date"] >= "2023-01-01"].copy()

# Renombrar columnas para que coincidan con tu app
df = df.rename(columns={
    "date": "fecha",
    "home_team": "local",
    "away_team": "visitante",
    "home_score": "goles_local",
    "away_score": "goles_visitante",
    "tournament": "competicion"
})

# Mantener únicamente las columnas necesarias
df = df[
    [
        "fecha",
        "competicion",
        "local",
        "visitante",
        "goles_local",
        "goles_visitante"
    ]
]

# Guardar el nuevo archivo
df.to_csv("partidos_2026.csv", index=False, encoding="utf-8")

print(f"✅ Dataset generado correctamente con {len(df)} partidos.")
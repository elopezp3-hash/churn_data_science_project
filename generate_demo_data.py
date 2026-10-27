"""Genera un dataset SINTÉTICO para comprobar la ejecución del código.
No representa clientes reales ni debe usarse para conclusiones académicas sobre el negocio."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000
uso = np.clip(rng.lognormal(mean=3.7, sigma=0.55, size=n), 1, 600)
antig = rng.integers(1, 73, n)
tickets = rng.poisson(2.5, n).astype(float)
edad = np.clip(rng.normal(42, 12, n), 18, 85)
monto = rng.choice([49, 99, 149, 249, 399], size=n, p=[.15,.25,.25,.2,.15]).astype(float)
genero = rng.choice(["Femenino", "Masculino", "Otro"], size=n, p=[.46,.50,.04])
region = rng.choice(["Caribe", "Andina", "Pacífica", "Orinoquía"], size=n, p=[.30,.45,.18,.07])
logit = -1.1 - 0.012*uso - 0.018*antig + 0.30*tickets + 0.002*monto
p = 1/(1+np.exp(-logit))
churn = rng.binomial(1, p)
df = pd.DataFrame({
    "ID_Cliente": [f"DEMO-{i:05d}" for i in range(n)],
    "Edad": edad.round().astype(int),
    "Antiguedad_Meses": antig.astype(str),
    "Soporte_Tickets": tickets,
    "Uso_Diario_Min": uso.round(1),
    "Monto_Mensual": monto,
    "Genero": genero,
    "Region": region,
    "Churn": churn
})
# Inserta algunos defectos sintéticos para probar la limpieza.
df.loc[3, "Edad"] = -5
df.loc[7, "Antiguedad_Meses"] = "no disponible"
df.loc[11, "Soporte_Tickets"] = np.nan
df.loc[15, "Uso_Diario_Min"] = 0
out = Path("data")
out.mkdir(exist_ok=True)
path = out / "demo_clientes_sinteticos.csv"
df.to_csv(path, index=False)
print(f"Dataset sintético de demostración guardado en {path}. NO son datos reales.")

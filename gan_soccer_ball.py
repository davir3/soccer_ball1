import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

# CONFIGURACION
RUTA = "dataset/balones/soccer_ball.npy"
SALIDA = "resultados"
EPOCAS = 10
LOTE = 128
RUIDO = 100
CANTIDAD = 20000

os.makedirs(SALIDA, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

print("======================================")
print(" PROYECTO GAN - SOCCER BALL")
print("======================================")

# CARGAR DATASET
datos = np.load(RUTA, mmap_mode="r")
cantidad = min(CANTIDAD, len(datos))

imagenes = np.array(datos[:cantidad], dtype=np.float32)
imagenes = imagenes / 127.5 - 1.0
imagenes = torch.tensor(imagenes)

cargador = DataLoader(
    TensorDataset(imagenes),
    batch_size=LOTE,
    shuffle=True,
    drop_last=True
)

print("Categoria: Soccer Ball")
print("Dibujos disponibles:", len(datos))
print("Dibujos utilizados:", cantidad)
print("Dimensiones:", imagenes.shape)

# GENERADOR
class Generador(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(RUIDO, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.red(x)

# DISCRIMINADOR
class Discriminador(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.red(x)

G = Generador()
D = Discriminador()

print("\nGENERADOR:")
print(G)
print("\nDISCRIMINADOR:")
print(D)

# PERDIDA Y OPTIMIZADORES
funcion_perdida = nn.BCEWithLogitsLoss()

opt_G = optim.Adam(
    G.parameters(), lr=0.0002, betas=(0.5, 0.999)
)

opt_D = optim.Adam(
    D.parameters(), lr=0.0002, betas=(0.5, 0.999)
)

historial_G = []
historial_D = []

# ENTRENAMIENTO
print("\nINICIANDO ENTRENAMIENTO...")

for epoca in range(EPOCAS):

    suma_G = 0
    suma_D = 0
    score_real = 0
    score_falso = 0

    for (reales,) in cargador:

        n = reales.size(0)

        etiquetas_reales = torch.ones(n, 1)
        etiquetas_falsas = torch.zeros(n, 1)

        # ENTRENAR DISCRIMINADOR
        ruido = torch.randn(n, RUIDO)
        falsas = G(ruido)

        salida_real = D(reales)
        salida_falsa = D(falsas.detach())

        loss_real = funcion_perdida(
            salida_real, etiquetas_reales
        )

        loss_falsa = funcion_perdida(
            salida_falsa, etiquetas_falsas
        )

        loss_D = (loss_real + loss_falsa) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ENTRENAR GENERADOR
        for p in D.parameters():
            p.requires_grad_(False)

        ruido = torch.randn(n, RUIDO)
        generadas = G(ruido)
        salida_G = D(generadas)

        loss_G = funcion_perdida(
            salida_G, etiquetas_reales
        )

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        for p in D.parameters():
            p.requires_grad_(True)

        suma_G += loss_G.item()
        suma_D += loss_D.item()

        score_real += torch.sigmoid(
            salida_real
        ).mean().item()

        score_falso += torch.sigmoid(
            salida_falsa
        ).mean().item()

    promedio_G = suma_G / len(cargador)
    promedio_D = suma_D / len(cargador)

    promedio_real = score_real / len(cargador)
    promedio_falso = score_falso / len(cargador)

    historial_G.append(promedio_G)
    historial_D.append(promedio_D)

    print(f"\nEPOCA {epoca+1}/{EPOCAS}")
    print(f"Perdida Generador: {promedio_G:.6f}")
    print(f"Perdida Discriminador: {promedio_D:.6f}")
    print(f"Puntuacion reales: {promedio_real:.6f}")
    print(f"Puntuacion falsas: {promedio_falso:.6f}")

# GENERAR IMAGENES
G.eval()
D.eval()

with torch.no_grad():

    ruido = torch.randn(64, RUIDO)
    generadas = G(ruido)

    pixeles = ((generadas + 1) * 127.5).clamp(0, 255)

    print("\nVALORES DE ENTRADA DEL GENERADOR:")
    print(ruido[0, :20].numpy())

    print("\nPRIMEROS 30 VALORES GENERADOS:")
    print(pixeles[0, :30].numpy())

    puntuaciones = torch.sigmoid(D(generadas))

    print("\nPUNTUACIONES DEL DISCRIMINADOR:")
    print(puntuaciones[:10].flatten().numpy())

    save_image(
        generadas.reshape(-1, 1, 28, 28),
        f"{SALIDA}/soccer_ball_generados.png",
        nrow=8,
        normalize=True,
        value_range=(-1, 1)
    )

    np.savetxt(
        f"{SALIDA}/valores_generados.csv",
        pixeles.numpy(),
        delimiter=","
    )

# GRAFICA DE PERDIDAS
plt.figure(figsize=(9, 5))
plt.plot(historial_G, label="Generador")
plt.plot(historial_D, label="Discriminador")
plt.xlabel("Epocas")
plt.ylabel("Perdida")
plt.title("GAN - Soccer Ball")
plt.legend()
plt.grid()
plt.savefig(f"{SALIDA}/grafica_perdidas.png")
plt.close()

# GUARDAR MODELOS
torch.save(G.state_dict(), f"{SALIDA}/generador.pth")
torch.save(D.state_dict(), f"{SALIDA}/discriminador.pth")

with open(f"{SALIDA}/perdidas.txt", "w") as f:
    for i, (g, d) in enumerate(zip(historial_G, historial_D)):
        f.write(
            f"Epoca {i+1}: Generador={g:.6f}, "
            f"Discriminador={d:.6f}\n"
        )

print("\n======================================")
print(" ENTRENAMIENTO FINALIZADO")
print(" RESULTADOS GUARDADOS EN:", SALIDA)
print("======================================")

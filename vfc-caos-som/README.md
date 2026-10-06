# Análisis topológico de la dinámica caótica en la VFC mediante Mapas Autoorganizados

Pipeline reproducible para diferenciar **ritmo sinusal normal (RSN)** de **insuficiencia cardíaca congestiva (ICC)** —y explorar el envejecimiento autonómico— a partir de intervalos R-R de PhysioNet, métricas de complejidad (Entropía Muestral, Entropía Multiescala, DFA, Poincaré, exponente de Lyapunov) y un **Mapa Autoorganizado (SOM)**.

## Notebooks (abrir en orden)

| # | Notebook | Qué hace | Responsable |
|---|---|---|---|
| 00 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Valoo-1201/vfc-caos-som/blob/main/notebooks/00_setup_colab_github.ipynb) `00_setup_colab_github` | Configura Colab, corre pruebas, sube cambios a GitHub | Todas |
| 01 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Valoo-1201/vfc-caos-som/blob/main/notebooks/01_extraccion_RR.ipynb) `01_extraccion_RR` | Descarga anotaciones, extrae intervalos NN, control de calidad | A |
| 02 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Valoo-1201/vfc-caos-som/blob/main/notebooks/02_metricas_no_lineales.ipynb) `02_metricas_no_lineales` | Tabla de métricas por ventana, estadística, sensibilidad de Lyapunov, sustitutos IAAFT | B |
| 03 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Valoo-1201/vfc-caos-som/blob/main/notebooks/03_SOM_agrupamiento.ipynb) `03_SOM_agrupamiento` | SOM, U-matriz, planos de componentes, k-means sobre prototipos | C |
| 04 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Valoo-1201/vfc-caos-som/blob/main/notebooks/04_evaluacion_validacion.ipynb) `04_evaluacion_validacion` | Validación por sujeto, ablación, bootstrap, eje de envejecimiento | C + B |


## Estructura

```
vfc-caos-som/
├── notebooks/            # solo orquestan y muestran resultados
├── src/hrvchaos/
│   ├── data.py           # lectura de PhysioNet con wfdb (+ datos sintéticos para pruebas)
│   ├── preprocess.py     # limpieza de NN y ventanas de 2000 latidos
│   ├── metrics.py        # SampEn, MSE, DFA, Poincaré, Lyapunov (Rosenstein), IAAFT
│   ├── som.py            # SOM hexagonal en NumPy: U-matriz, QE, TE, mapas de impactos
│   ├── evaluate.py       # tabla de características, estadística, validación agrupada por sujeto
│   └── plots.py          # figuras
├── tests/test_metrics.py # las métricas reproducen valores conocidos (ruido blanco, mapa logístico…)
└── results/              # CSV y figuras pequeñas (sí se versionan)
```

Los datos crudos y la caché de series R-R se guardan en Google Drive (`MyDrive/vfc-caos-som-data`), **no** en el repositorio.

## Bases de datos (PhysioNet, licencia ODC-By)

| Base | Grupo | Sujetos | fs |
|---|---|---|---|
| `nsr2db` | RSN | 54 | 128 Hz |
| `chf2db` | ICC (NYHA I–III) | 29 | 128 Hz |
| `chfdb` | ICC (NYHA III–IV), validación externa | 15 | 250 Hz |
| `nsrdb` | RSN, validación externa | 18 | 128 Hz |
| `fantasia` | Jóvenes / mayores sanos | 20 + 20 | 250 Hz |

## Flujo de trabajo en equipo

1. Abrir el notebook con el botón *Colab* y ejecutar la **celda 0** (clona/actualiza el repo, instala dependencias, monta Drive).
2. Trabajar en una rama propia: `feat/<tema>`.
3. La lógica reutilizable va en `src/hrvchaos/`; el notebook solo la llama.
4. Antes de subir: *Reiniciar y ejecutar todo*.
5. Subir con la última celda de `00_setup_colab_github` (usa el secreto `GITHUB_TOKEN` de Colab) y abrir un *pull request*; otra compañera lo revisa.

Reglas: semilla fija (`SEED = 42`); nunca escribir tokens en celdas; validación siempre **agrupada por sujeto**.

## Probar sin internet

Pon `MODO_DEMO = True` en la celda 0: el pipeline corre con series sintéticas (no fisiológicas) para comprobar que el código funciona.

## Referencias principales

Goldberger et al. 2000 (PhysioNet) · Richman & Moorman 2000 (SampEn) · Costa et al. 2002 (MSE) · Peng et al. 1995 (DFA) · Rosenstein et al. 1993 (Lyapunov) · Brennan et al. 2001 (Poincaré) · Kohonen 1990 (SOM) · Vesanto & Alhoniemi 2000 (clustering del SOM) · Schreiber & Schmitz 1996 (IAAFT).

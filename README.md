# 👟 Visualizador de Métricas — Detector de Zapatos con YOLOv8

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge&logo=pytorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Compatible-yellow?style=for-the-badge&logo=googlecolab&logoColor=white)

*Herramienta de visualización de métricas de rendimiento para el modelo de detección de calzado entrenado con YOLOv8*

</div>

---

## 👨‍💻 Autores

| Nombre | Matrícula | Grupo |
|--------|-----------|-------|
| **Rubén Dario Ibarra Tirado** | 23310353 | 6F |
| **José Alberto Castañeda Sánchez** | 23310352 | 6F |

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Caso de Estudio](#-caso-de-estudio)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [¿Cómo Funciona?](#-cómo-funciona)
- [Solución de Problemas](#-solución-de-problemas)

---

## 📌 Descripción

Este script permite **visualizar de forma clara y profesional** las métricas de rendimiento generadas automáticamente por YOLOv8 al finalizar el entrenamiento de nuestro **detector de zapatos**. En lugar de acceder manualmente a los archivos de salida dentro del directorio `runs/`, este visualizador centraliza y despliega la gráfica de resultados con una presentación limpia, sin ejes de píxeles y con un título descriptivo.

Es especialmente útil en entornos como **Google Colab**, donde los archivos se generan en rutas dinámicas (`train`, `train-2`, `train-3`, etc.) y se necesita una verificación rápida del rendimiento del modelo de detección de calzado.

---

## 🔬 Caso de Estudio

### Detección de Zapatos con Visión Artificial: ¿Por qué importa?

La industria del calzado mueve millones de pares al año a través de cadenas de producción, almacenes, puntos de venta y plataformas de comercio electrónico. En este contexto, contar con un sistema capaz de **detectar y reconocer zapatos automáticamente en imágenes o video en tiempo real** abre la puerta a múltiples aplicaciones de alto valor:

- **Inventario automatizado** en tiendas y almacenes, eliminando conteos manuales.
- **Clasificación y etiquetado automático** de productos en plataformas de e-commerce.
- **Sistemas de seguridad y análisis de comportamiento** en puntos de venta (detección de clientes, tráfico de productos).
- **Control de calidad** en líneas de producción de calzado para identificar defectos o pares incompletos.
- **Prueba virtual de calzado** como componente previo en aplicaciones de Realidad Aumentada.

### El Reto del Entrenamiento

Entrenar un modelo YOLOv8 para detectar zapatos no es trivial. El modelo debe aprender a distinguir zapatos en distintas condiciones: diferentes ángulos, iluminación, colores, estilos (tenis, botas, sandalias, tacones) y contextos (sobre una persona, en aparador, en caja). Esto hace que el proceso de entrenamiento sea iterativo y que **monitorear las métricas sea indispensable** para saber si el modelo realmente está aprendiendo a detectar calzado de forma generalizada o simplemente memorizando ejemplos.

Sin una revisión adecuada de las métricas, es imposible determinar si el modelo:

- Está **convergiendo** correctamente y aprendiendo a localizar zapatos.
- Presenta **sobreajuste (overfitting)**: detecta bien en el dataset de entrenamiento pero falla con zapatos nuevos.
- Sufre de **subajuste (underfitting)**: no logra aprender las características visuales del calzado.
- Requiere **ajuste de hiperparámetros** como la tasa de aprendizaje, el número de épocas o el tamaño del batch.

YOLOv8 genera automáticamente un archivo `results.png` con las siguientes métricas clave al finalizar cada entrenamiento:

| Métrica | Descripción |
|--------|-------------|
| `train/box_loss` | Pérdida en la predicción de bounding boxes (entrenamiento) |
| `train/cls_loss` | Pérdida en la clasificación de clases (entrenamiento) |
| `train/dfl_loss` | Pérdida de distribución focal (entrenamiento) |
| `val/box_loss` | Pérdida en bounding boxes sobre datos de validación |
| `val/cls_loss` | Pérdida en clasificación sobre datos de validación |
| `val/dfl_loss` | Pérdida de distribución focal sobre validación |
| `metrics/precision(B)` | Precisión del modelo sobre el conjunto de validación |
| `metrics/recall(B)` | Exhaustividad del modelo sobre el conjunto de validación |
| `metrics/mAP50(B)` | Mean Average Precision al umbral IoU 0.50 |
| `metrics/mAP50-95(B)` | mAP promediado entre umbrales IoU 0.50 a 0.95 |

### ¿En qué nos beneficia esta herramienta para nuestro detector de zapatos?

1. **Evaluación rápida del aprendizaje:** Permite ver de un vistazo si el modelo está aprendiendo correctamente a detectar diferentes tipos de calzado a lo largo de las épocas.
2. **Detección temprana de problemas:** Si la pérdida de validación sube mientras la de entrenamiento baja, es señal de que el modelo está memorizando zapatos del dataset y no generalizando.
3. **Ahorro de tiempo:** Evita navegar manualmente entre carpetas para encontrar el archivo correcto de resultados tras cada sesión de entrenamiento.
4. **Flexibilidad entre sesiones:** Detecta automáticamente si la carpeta de resultados es `train`, `train-2`, `train-3`, etc., algo común cuando se experimenta con distintas configuraciones del detector.
5. **Documentación del progreso:** Facilita capturar y comparar el rendimiento entre distintos experimentos, por ejemplo al variar el dataset de zapatos o los hiperparámetros.

---

## ⚙️ Requisitos

Asegúrate de tener instaladas las siguientes dependencias antes de ejecutar el script:

```
Python >= 3.8
matplotlib >= 3.5.0
ultralytics (YOLOv8) >= 8.0.0
```

También necesitas haber entrenado previamente un modelo con YOLOv8 para que exista la carpeta de resultados.

---

## 🚀 Instalación

### 1. Clona o descarga este repositorio

```bash
git clone https://github.com/tu-usuario/yolov8-metrics-viewer.git
cd yolov8-metrics-viewer
```

### 2. Instala las dependencias necesarias

```bash
pip install matplotlib ultralytics
```

### 3. (Opcional) En Google Colab

```python
!pip install ultralytics
```

---

## 🖥️ Uso

### Paso 1 — Entrenamiento del modelo

Asegúrate de haber ejecutado un entrenamiento con YOLOv8 previamente:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=50, imgsz=640)
```

Esto generará automáticamente la carpeta `runs/detect/train/` (o `train-2`, `train-3`, etc.) con el archivo `results.png`.

### Paso 2 — Ejecutar el visualizador

```python
import matplotlib.pyplot as plt
import os

# Ajusta la ruta según el número de entrenamiento actual
path_resultados = 'runs/detect/train-2/results.png'

if os.path.exists(path_resultados):
    img = plt.imread(path_resultados)
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Métricas de Rendimiento del Entrenamiento (YOLOv8)", fontsize=14, fontweight='bold')
    plt.show()
else:
    print(f"No se encontró el archivo de resultados en la ruta: {path_resultados}")
    print("Asegúrate de verificar si la carpeta es 'train', 'train-2', etc.")
```

> **💡 Tip:** Si no sabes el número de tu sesión de entrenamiento, puedes listar las carpetas disponibles con:
> ```python
> import os
> carpetas = os.listdir('runs/detect/')
> print(carpetas)
> ```

---

## 📈 Resultados

A continuación se muestra un ejemplo representativo de las métricas generadas por YOLOv8 durante el entrenamiento y visualizadas con este script:

### Gráfica de Pérdidas y Métricas

```
runs/detect/train-2/results.png
```

> *La imagen generada por YOLOv8 incluye las siguientes subgráficas en una cuadrícula:*

```
┌──────────────────────────────────────────────────────────────────┐
│  train/box_loss  │  train/cls_loss  │  train/dfl_loss            │
├──────────────────────────────────────────────────────────────────┤
│  val/box_loss    │  val/cls_loss    │  val/dfl_loss              │
├──────────────────────────────────────────────────────────────────┤
│  precision(B)    │  recall(B)       │  mAP50(B)  │  mAP50-95(B) │
└──────────────────────────────────────────────────────────────────┘
```

**Ejemplo de visualización final:**

![Ejemplo de métricas YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

> *Imagen referencial. Tu gráfica `results.png` se generará específicamente con los datos de tu entrenamiento.*

### Interpretación de los Resultados

| Indicador | Valor Ideal | Interpretación |
|-----------|-------------|----------------|
| `box_loss` | Decreciente | El modelo mejora en localizar objetos |
| `cls_loss` | Decreciente | El modelo mejora en clasificar objetos |
| `mAP50` | > 0.80 | Buena detección al umbral IoU del 50% |
| `mAP50-95` | > 0.60 | Detección robusta en múltiples umbrales |
| `precision` | > 0.85 | Pocos falsos positivos |
| `recall` | > 0.85 | Pocos falsos negativos |

---

## 🗂️ Estructura del Proyecto

```
yolov8-metrics-viewer/
│
├── visualizer.py             # Script principal de visualización
├── README.md                 # Este archivo
│
└── runs/
    └── detect/
        ├── train/            # Primera sesión de entrenamiento
        │   ├── results.png   ← Gráfica de métricas
        │   ├── weights/
        │   │   ├── best.pt
        │   │   └── last.pt
        │   └── ...
        └── train-2/          # Segunda sesión (la utilizada en este script)
            ├── results.png   ← Gráfica de métricas
            └── ...
```

---

## 🔍 ¿Cómo Funciona?

```
┌─────────────────────────────────────────────────────┐
│                  FLUJO DEL SCRIPT                   │
└─────────────────────────────────────────────────────┘

  1. Define la ruta al archivo results.png
        │
        ▼
  2. Verifica si el archivo existe (os.path.exists)
        │
        ├── ✅ SÍ existe ──► Lee la imagen con plt.imread()
        │                         │
        │                         ▼
        │                    Crea figura (12x10 pulgadas)
        │                         │
        │                         ▼
        │                    Muestra imagen sin ejes
        │                         │
        │                         ▼
        │                    Agrega título descriptivo
        │                         │
        │                         ▼
        │                    plt.show() → Renderiza
        │
        └── ❌ NO existe ──► Imprime mensaje de error
                              con ruta intentada y
                              sugerencia de corrección
```

---

## 🛠️ Solución de Problemas

| Problema | Causa probable | Solución |
|---------|---------------|----------|
| `No se encontró el archivo...` | Ruta incorrecta | Revisa el número de sesión (`train`, `train-2`, etc.) |
| La imagen se ve muy pequeña | `figsize` muy pequeño | Aumenta a `figsize=(16, 12)` |
| Error al importar matplotlib | Librería no instalada | Ejecuta `pip install matplotlib` |
| La imagen está en blanco | Entrenamiento incompleto | Espera a que finalicen al menos 1 época |
| Carpeta `runs/` no encontrada | No se ha entrenado aún | Ejecuta primero el entrenamiento del detector de zapatos con `model.train()` |
| `mAP50` muy bajo (< 0.4) | Dataset de zapatos insuficiente o mal etiquetado | Revisa las anotaciones y considera aumentar el dataset con más variedad de calzado |
| Pérdida de validación sube | Sobreajuste al dataset de zapatos | Reduce épocas, aplica data augmentation o amplía el dataset |

---

## 📚 Referencias

- [Documentación oficial de Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Google Colab — Entornos de entrenamiento](https://colab.research.google.com/)
- Redmon, J. & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement.* arXiv:1804.02767
- Jocher, G. et al. (2023). *Ultralytics YOLOv8.* GitHub Repository.

---

<div align="center">

**CETI Colomos - Ingeniería en Mecatrónica · Grupo 6F · 2025**

*Desarrollado con 💻 por Dario Ibarra Tirado y José Alberto Castañeda Sánchez*

</div>

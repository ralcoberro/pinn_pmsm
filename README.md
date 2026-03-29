# Modelo Sustituto de Torque basado en PANN para Motores PMSM

## Descripción general

Este proyecto implementa un modelo sustituto (*surrogate model*) de torque para motores sincrónicos de imanes permanentes (PMSM) de imanes superficiales, mucleo ferromagnético y rotor externo, combinando simulaciones por elementos finitos (FEA) con redes neuronales asistidas por la física (PANN — *Physics-Assisted Neural Network*). El objetivo es reemplazar las costosas evaluaciones FEA durante procesos de optimización multi-objetivo del diseño del motor, manteniendo una precisión aceptable con una cantidad reducida de datos de entrenamiento.

### Modelo de diseño paramétrico y compresión de la información

El diseño de los motores PMSM se realiza sobre un plano de diseño paramétrico *(l, b)* propuesto en [1]. El parámetro *b = Bg₁ / Bh_max* relaciona la densidad de flujo magnético fundamental en el entrehierro con la máxima densidad de flujo permitida en el material ferromagnético, gobernando proporciones geométricas clave como el ancho de diente, el espesor del imán y el espesor del material ferromangnétco del rotor *(backiron)*. El parámetro *l = (Ro − Ri) / Ro* indica la proporción de la región electromagnéticamente activa respecto al radio exterior total, definiendo si la máquina tiene geometría anular (*l* ≤ 0.5) o tipo disco (*l* > 0.5). 
Esta parametrización comprime la información del espacio de diseño, que originalmente involucra una gran cantidad de variables geométricas y electromagnéticas interdependientes, en tan solo dos parámetros adimensionales, lo que permite explorar de manera eficiente y sistemática todo el espacio de diseño factible, identificar frentes de Pareto óptimos entre torque específico y eficiencia, y generar familias completas de diseños a partir de un modelo analítico compacto. Esta compresión es fundamental para la viabilidad del enfoque de aprendizaje automático, ya que reduce drásticamente la dimensionalidad de las entradas de la red neuronal y, por ende, la cantidad de simulaciones FEA necesarias para el entrenamiento.
NOTA: En el notebook se utiliza la letra *x* en lugar de *l*, pero su significado es el mismo.

### Uso de la transformada de Fourier para reducir la dimensionalidad de la salida

Un aspecto clave del enfoque adoptado es que la red neuronal no aprende directamente la forma de onda temporal completa del torque (que requeriría decenas o cientos de puntos por ciclo eléctrico), sino su **descomposición en armónicos de Fourier**. Esto permite representar la señal completa de torque, incluyendo el valor medio (DC) y el ripple, con un número muy reducido de salidas (amplitud y fase de unos pocos armónicos seleccionados), reduciendo significativamente la cantidad de neuronas necesarias en la capa de salida de la red y simplificando el problema de aprendizaje. Este enfoque está inspirado en el trabajo de Song et al. [2], donde se propone utilizar la representación en el dominio de Fourier tanto para las entradas como para las salidas de la red neuronal, mejorando la eficiencia del entrenamiento y la precisión de la predicción del torque y su rizado.

### Asistencia física para el 6to armónico: cogging torque analítico

Entre los armónicos seleccionados (`SELECTED_HARMONICS = [0, 2, 6]`), el **6to armónico** resulta de particular interés y dificultad. Para la combinación de 18 ranuras y 24 polos (12 pares de polos) utilizada en este trabajo, el número de períodos de cogging por revolución mecánica es N_c = LCM(18, 24) = 72. Como cada revolución eléctrica corresponde a 1/12 de la revolución mecánica, se producen 72/12 = 6 períodos de cogging por ciclo eléctrico. Es decir, el **cogging torque se manifiesta precisamente como el 6to armónico** de la forma de onda de torque en el dominio eléctrico.

El cogging torque es un fenómeno altamente no lineal y muy sensible a variaciones geométricas, lo que lo hace difícil de predecir tanto para modelos analíticos como para redes neuronales puramente basadas en datos. Para asistir a la red en esta tarea, se implementa el enfoque PANN [2]: se calcula analíticamente la amplitud pico del cogging torque mediante el **método analítico de Zhu y Howe** [3], que estima el cogging torque a partir de la distribución de densidad de flujo magnético (expansión en armónicos espaciales) y la función de permeancia del entrehierro (que incorpora el efecto de las aperturas de ranura mediante el coeficiente de Carter). Este valor obtenido analíticamente (`cogging_zhu`) se agrega como una **6ta entrada** a la red neuronal, proporcionando información física directa sobre el fenómeno que gobierna el 6to armónico. De esta forma, la red no necesita aprender el cogging torque desde cero, sino que aprende a *corregir* la estimación analítica, lo que permite alcanzar mayor precisión con menos datos de entrenamiento.

## Notebooks

### 1. `generate_pinn_dataset.ipynb` — Generación del dataset

Este notebook genera el dataset de diseños de motores y sus correspondientes formas de onda de torque evaluadas por FEA. El proceso se divide en dos partes:

**Parte 1 — Generación de diseños:**
- Muestrea el espacio de diseño paramétrico *(b, l, outer_radius)* mediante *Latin Hypercube Sampling* (LHS).
- Para cada muestra, construye el motor PMSM utilizando el modelo analítico (clase `PMSM` de `arfemm`).
- Genera el archivo de modelo `.fem` para cada diseño válido en la herramienta FEMM.
- Calcula indicadores de desempeño (torque, eficiencia, masa, etc.) a partir del modelo analítico.
- Guarda los resultados en `dataset/designs.csv` y los archivos FEM en `dataset/designs/`.

**Parte 2 — Barrido de torque por FEA:**
- Para cada modelo FEM válido, busca el ángulo de conmutación óptimo (máximo torque para id=0).
- Realiza un barrido de 360° eléctricos, calculando el torque electromagnético en cada posición angular mediante FEMM.
- Calcula la FFT de la forma de onda de torque y almacena los primeros 10 armónicos (amplitud y fase).
- Guarda los resultados en `dataset/torque_results.csv` y las formas de onda individuales en `dataset/torque_waveforms/`.

**Estructura de salida:**
```
dataset/
    designs.csv              # Parámetros de diseño y resultados analíticos
    torque_results.csv       # Armónicos de torque (de FEA)
    designs/
        model0001.fem        # Archivos de modelo FEMM
        model0002.fem
        ...
    torque_waveforms/
        model0001.csv        # Forma de onda de torque completa
        ...
```

### 2. `pinn_torque_femm_pytorch.ipynb` — Entrenamiento del modelo sustituto

Este notebook entrena una red neuronal MLP (*Multi-Layer Perceptron*) en PyTorch que actúa como modelo sustituto del torque.

**Pipeline:**
1. **Carga de datos:** une `designs.csv` con `torque_results.csv` por `sample_id`.
2. **Entradas (5 o 6):** parámetros de diseño `b`, `x`, `outer_radius`, `phase_current`, `torque` (analítico). Opcionalmente se agrega `cogging_zhu` (torque de cogging calculado por método analítico de Zhu-Howe) como 6ª entrada, lo que implementa el enfoque PANN: el modelo analítico provee una estimación inicial que la red aprende a *corregir*.
3. **Salidas:** amplitud y fase de los armónicos seleccionados de Fourier (configurable mediante `SELECTED_HARMONICS`, por defecto `[0, 2, 6]`).
4. **Normalización:** inputs y outputs escalados a `[-1, 1]` mediante min-max.
5. **Arquitectura:** MLP configurable (por defecto, 3-6 capas ocultas de 32-128 neuronas con activación Tanh).
6. **Entrenamiento:** pérdida MSE, optimizador Adam, scheduler (ReduceLROnPlateau o CosineAnnealingWarmRestarts), early stopping opcional.
7. **Evaluación:** métricas R² por armónico, comparación de formas de onda reconstruidas (predicha vs. FEA).
8. **Guardado:** checkpoint `.pt` con pesos del modelo, normalizadores y metadata.

## Ejecución

### Requisitos previos

1. **Python 3.10+** con las dependencias listadas abajo.
2. **FEMM 4.2** — herramienta de análisis por elementos finitos para electromagnetismo. Instalar desde [www.femm.info](http://www.femm.info). Necesario únicamente para el notebook de generación de dataset.
3. **KOIL** — software para el cálculo del esquema de bobinado del motor. Necesario para la construcción de los modelos FEM, únicamente para el notebook de generación de dataset.
4. **Biblioteca `arfemm`** — el módulo Python `pmsm.py` y `femm_utils.py` ubicados en `arfemm/lib/python/`. El path al módulo debe configurarse correctamente (por defecto apunta a `c:\Proyectos\arfemm\lib\python`). `arfemm` es una biblioteca en Matlab y Python, realizada por Rodrigo Alcoberro, para el análisis y diseño de PMSM.

### Paso 1: Generar el dataset

```bash
jupyter notebook generate_pinn_dataset.ipynb
```

Ejecutar todas las celdas en orden. Esto:
- Genera las muestras LHS del espacio de diseño.
- Construye cada motor en FEMM (requiere que FEMM esté instalado y accesible).
- Ejecuta los barridos de torque por FEA.
- Produce los archivos CSV con los resultados.

> **Nota:** Este paso requiere un entorno Windows con FEMM instalado y puede llevar varias horas, dependiendo del número de muestras (`N_SAMPLES`).

### Paso 2: Entrenar el modelo sustituto

```bash
jupyter notebook pinn_torque_femm_pytorch.ipynb
```

Ejecutar todas las celdas en orden. El notebook:
- Carga el dataset generado en el paso anterior.
- Entrena la red neuronal (se recomienda GPU para mayor velocidad).
- Evalúa el modelo y muestra métricas/gráficos comparativos.
- Guarda el checkpoint del modelo entrenado en `model/`.

> **Nota:** Si el dataset ya fue generado previamente, este notebook puede ejecutarse de forma independiente. Si ya existe un modelo entrenado, se puede cargar directamente omitiendo las celdas de entrenamiento.

## Dependencias adicionales

| Paquete           | Uso                                                    |
| :---------------- | :----------------------------------------------------- |
| `torch`           | Framework de aprendizaje profundo (entrenamiento MLP)  |
| `numpy`           | Operaciones numéricas y FFT                            |
| `pandas`          | Manipulación de datos tabulares (CSV)                  |
| `matplotlib`      | Visualización de resultados                            |
| `scikit-learn`    | Split train/test, métricas (R²)                        |
| `tqdm`            | Barras de progreso durante entrenamiento               |
| `scipy`           | Latin Hypercube Sampling (`scipy.stats.qmc`)           |
| `femm`            | Interfaz Python para FEMM (solo generación de dataset) |

Instalación de las dependencias Python:
```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm scipy
```

El paquete `femm` se instala automáticamente con la instalación de FEMM 4.2 en Windows.

## Referencias

[1] R. Alcoberro, S. Gonzalez, and S. Verne, "Pareto-Optimal Parametric Design Method for PMSM Applied to eVTOL UAV," in *Proc. VI International Conference on Electrical, Computer and Energy Technologies (ICECET 2026)*, Rome, Italy, Jul. 2026 (pre-print, no publicado aún. CONFIDENCIAL).

[2] J. Song, Y.-W. Chen and K. Hameyer, "Multi-Objective Motor Design Optimization with Physics-Assisted Neural Network Model," in *2023 IEEE International Electric Machines & Drives Conference (IEMDC)*, San Francisco, CA, USA, 2023. DOI: [10.1109/IEMDC55163.2023.10238886](https://ieeexplore.ieee.org/document/10238886)

[3] Z. Q. Zhu and D. Howe, "Influence of design parameters on cogging torque in permanent magnet machines," *IEEE Transactions on Energy Conversion*, vol. 15, no. 4, pp. 407–412, Dec. 2000. DOI: [10.1109/60.900502](https://doi.org/10.1109/60.900502)

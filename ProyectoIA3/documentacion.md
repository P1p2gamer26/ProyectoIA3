# Proyecto 3 — Motor de Inferencia por Enumeración
**Introducción a la Inteligencia Artificial · 6.º semestre**

---

## Tabla de contenido

1. [Descripción del problema](#1-descripción-del-problema)
2. [Fundamento teórico](#2-fundamento-teórico)
   - 2.1 [Variables aleatorias y dominio](#21-variables-aleatorias-y-dominio)
   - 2.2 [Probabilidad condicional](#22-probabilidad-condicional)
   - 2.3 [Red Bayesiana](#23-red-bayesiana)
   - 2.4 [Inferencia probabilística](#24-inferencia-probabilística)
   - 2.5 [Algoritmo de enumeración](#25-algoritmo-de-enumeración)
3. [Diseño de clases](#3-diseño-de-clases)
4. [Archivos de datos](#4-archivos-de-datos)
5. [Algoritmo paso a paso](#5-algoritmo-paso-a-paso)
6. [Resultados y validación](#6-resultados-y-validación)
7. [Cómo ejecutar](#7-cómo-ejecutar)

---

## 1. Descripción del problema

El proyecto implementa un **Motor de Inferencia por Enumeración** sobre Redes Bayesianas. El dominio de demostración es el ejemplo del *Tren y la Reunión*, visto en clase:

- Una variable de **lluvia** (`Rain`) influye en si hay **mantenimiento** en las vías (`Maintenance`).
- Ambas influyen en si el **tren** llega a tiempo o se retrasa (`Train`).
- El estado del tren determina si se puede **asistir a la reunión** (`Appointment`).

```
Rain ──────────────────────────────────┐
  │                                    ▼
  └──► Maintenance ──────────► Train ──► Appointment
```

El motor puede responder preguntas como:
> *"Dada lluvia ligera y sin mantenimiento, ¿cuál es la probabilidad de llegar a la reunión?"*

---

## 2. Fundamento teórico

### 2.1 Variables aleatorias y dominio

Una **variable aleatoria** tiene un conjunto finito de valores posibles llamado **dominio**. En nuestra red:

| Variable | Dominio |
|----------|---------|
| `Rain` | `{none, light, heavy}` |
| `Maintenance` | `{yes, no}` |
| `Train` | `{on_time, delayed}` |
| `Appointment` | `{attend, miss}` |

### 2.2 Probabilidad condicional

La **probabilidad condicional** `P(a | b)` representa la probabilidad de que `a` ocurra dado que sabemos que `b` ya ocurrió:

```
P(a | b) = P(a ∧ b) / P(b)
```

Reordenando obtenemos la **regla del producto**:

```
P(a ∧ b) = P(a | b) · P(b)
```

Esta regla se generaliza para calcular probabilidades conjuntas:

```
P(A, B, C, D) = P(A) · P(B|A) · P(C|A,B) · P(D|C)
```

### 2.3 Red Bayesiana

Una **Red Bayesiana** es un Grafo Dirigido Acíclico (DAG) donde:

- Cada **nodo** representa una variable aleatoria.
- Cada **arco** `X → Y` significa que `Y` depende directamente de `X`.
- Cada nodo `X` almacena la distribución **P(X | padres(X))** en su Tabla de Probabilidad Condicional (CPT).

Las CPTs de la red del Tren y la Reunión son:

**Rain** (sin padres — probabilidad a priori):

| none | light | heavy |
|------|-------|-------|
| 0.7  | 0.2   | 0.1   |

**Maintenance** (condicionada a Rain):

| Rain  | yes | no  |
|-------|-----|-----|
| none  | 0.4 | 0.6 |
| light | 0.2 | 0.8 |
| heavy | 0.1 | 0.9 |

**Train** (condicionada a Rain y Maintenance):

| Rain  | Maintenance | on_time | delayed |
|-------|-------------|---------|---------|
| none  | yes         | 0.8     | 0.2     |
| none  | no          | 0.9     | 0.1     |
| light | yes         | 0.6     | 0.4     |
| light | no          | 0.7     | 0.3     |
| heavy | yes         | 0.4     | 0.6     |
| heavy | no          | 0.5     | 0.5     |

**Appointment** (condicionada a Train):

| Train   | attend | miss |
|---------|--------|------|
| on_time | 0.9    | 0.1  |
| delayed | 0.6    | 0.4  |

### 2.4 Inferencia probabilística

Dado un estado parcialmente observado del mundo, queremos calcular la distribución de probabilidad de una variable desconocida.

Se distinguen tres tipos de variables:

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **Variable consulta X** | La que queremos calcular | `Appointment` |
| **Variables evidencia E = e** | Observadas directamente | `Rain=light, Maintenance=no` |
| **Variables ocultas Y** | Ni consulta ni evidencia | `Train` |

**Objetivo:** calcular `P(X | e)`.

Usando la definición de probabilidad condicional y la regla del producto:

```
P(X | e) = α · P(X, e) = α · Σ_Y P(X, e, Y)
```

donde `α = 1 / Σ_xi P(xi, e)` es el **factor de normalización** que garantiza que las probabilidades sumen 1.

### 2.5 Algoritmo de enumeración

El algoritmo **ENUMERATION-ASK** (Russell & Norvig, Cap. 13) calcula `P(X | e)` recorriendo las variables en **orden topológico**:

```
ENUMERATION-ASK(X, e, bn):
  para cada valor xi de X:
    Q(xi) ← ENUMERATE-ALL(bn.VARS, e ∪ {X = xi})
  retornar NORMALIZE(Q)

ENUMERATE-ALL(vars, e):
  si vars está vacío: retornar 1.0
  Y ← primera variable de vars
  si Y tiene valor y en e:
    retornar P(y | padres(Y)) × ENUMERATE-ALL(resto, e)
  si no (Y es oculta):
    retornar Σ_y  P(y | padres(Y)) × ENUMERATE-ALL(resto, e ∪ {Y = y})
```

**¿Por qué orden topológico?** En un DAG, el orden topológico garantiza que los padres de cada nodo siempre son procesados *antes* que el nodo mismo. Así, cuando el algoritmo necesita calcular `P(Y | padres(Y))`, los valores de los padres ya están disponibles en la evidencia acumulada.

---

## 3. Diseño de clases

El código está organizado en cuatro clases en `bayesian_network.py`, siguiendo el principio de responsabilidad única:

```
┌────────────────────────────────────────────────────────────┐
│                      BayesianNetwork                       │
│  + nodes: dict[str → Node]                                 │
│  + arcs:  list[Arc]                                        │
│  ─────────────────────────────────────────────────────     │
│  + load_structure(filepath)                                │
│  + load_probabilities(filepath)                            │
│  + display_structure()                                     │
│  + display_probabilities()                                 │
│  + enumerate_ask(query, evidence, trace) → dict            │
│  + _enumerate_all(vars, evidence, trace, depth) → float    │
│  + topological_sort() → list[Node]                         │
└──────────────────────────┬─────────────────────────────────┘
                           │ contiene
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
     ┌───────────┐   ┌───────────┐   ┌───────────┐
     │   Node    │   │    Arc    │   │    CPT    │
     │ + name    │   │ + source  │   │ + entries │
     │ + values  │   │ + dest    │   │ + values  │
     │ + parents │   └───────────┘   │           │
     │ + children│                   │ + add_entry│
     │ + cpt     │ ──── tiene ──────►│ + get_prob│
     └───────────┘                   └───────────┘
```

### `CPT` — Tabla de Probabilidad Condicional

Almacena internamente un diccionario:

```python
entries: { (val_padre1, val_padre2, ...) → { val_nodo: probabilidad } }
```

La clave es una **tupla** con los valores de los padres en el orden en que fueron declarados en el archivo, lo que garantiza consistencia en la búsqueda.

### `Node` — Variable aleatoria

Encapsula el dominio de la variable y sus relaciones de dependencia. El método `get_probability(value, parent_assignment)` convierte el diccionario `{nombre: valor}` en la tupla que espera la CPT, respetando el orden declarado.

### `Arc` — Arco dirigido

Registra la relación de dependencia `padre → hijo`. Se usa principalmente para contabilizar los arcos de la red y para visualización.

### `BayesianNetwork` — Motor principal

Contiene el grafo completo y el algoritmo de inferencia. Los métodos de carga (`load_structure`, `load_probabilities`) son independientes del dominio: la red puede representar cualquier problema con solo cambiar los archivos de datos.

---

## 4. Archivos de datos

El sistema es **genérico**: la red se define completamente por dos archivos de texto, sin necesidad de modificar el código Python.

### `network_structure.txt`

Define los arcos del grafo. Cada línea especifica una relación causal:

```
# Formato: nodo_padre,nodo_hijo
Rain,Maintenance
Rain,Train
Maintenance,Train
Train,Appointment
```

### `network_probabilities.txt`

Define las CPTs. Cada bloque corresponde a un nodo:

```
# Nodo sin padres (una sola fila):
NODE Rain
VALUES none light heavy
0.7 0.2 0.1

# Nodo con padres (una fila por combinación):
NODE Train
VALUES on_time delayed
PARENTS Rain Maintenance
none  yes  0.8 0.2
none  no   0.9 0.1
...
```

**Reglas de formato:**
- Las líneas que comienzan con `#` son comentarios y se ignoran.
- Las líneas en blanco se ignoran.
- El orden de las columnas de padres en cada fila debe coincidir con el orden declarado en `PARENTS`.
- Las probabilidades de cada fila deben sumar 1.0.

---

## 5. Algoritmo paso a paso

### Ejemplo de la clase

**Consulta:** `P(Appointment | Rain=light, Maintenance=no)`

**Identificación de roles:**
- Variable consulta: `Appointment` con dominio `{attend, miss}`
- Evidencia: `{Rain=light, Maintenance=no}`
- Variable oculta: `Train`
- Orden topológico: `Rain → Maintenance → Train → Appointment`

---

**Paso 1:** Calcular el término para `Appointment = attend`

Extendemos la evidencia: `{Rain=light, Maintenance=no, Appointment=attend}`

`ENUMERATE-ALL([Rain, Maintenance, Train, Appointment], e_extendida)`

```
Rain=light  ∈ evidencia  → P(light) = 0.2
  Maintenance=no ∈ evidencia  → P(no | light) = 0.8
    Train es oculta → sumar sobre {on_time, delayed}:
      Train=on_time:
        P(on_time | light, no) = 0.7
          Appointment=attend ∈ evidencia → P(attend | on_time) = 0.9
          subtotal = 0.7 × 0.9 = 0.630
      Train=delayed:
        P(delayed | light, no) = 0.3
          Appointment=attend ∈ evidencia → P(attend | delayed) = 0.6
          subtotal = 0.3 × 0.6 = 0.180
    suma Train = 0.630 + 0.180 = 0.810
  resultado = 0.2 × 0.8 × 0.810 = 0.1296
```

**Paso 2:** Calcular el término para `Appointment = miss`

```
0.2 × 0.8 × (0.7×0.1 + 0.3×0.4) = 0.2 × 0.8 × 0.190 = 0.0304
```

**Paso 3:** Normalizar

```
α = 1 / (0.1296 + 0.0304) = 1 / 0.16 = 6.25

P(attend | light, no) = 0.1296 × 6.25 = 0.81
P(miss   | light, no) = 0.0304 × 6.25 = 0.19
```

---

## 6. Resultados y validación

### Parte 1 — Probabilidades conjuntas (verificación de carga)

| Caso | Cálculo | Resultado | Esperado |
|------|---------|-----------|----------|
| Rain=light, Maint=no, Train=delayed, Appt=miss | 0.2 × 0.8 × 0.3 × 0.4 | **0.0192** | 0.0192 ✓ |
| Rain=heavy, Maint=yes, Train=delayed, Appt=attend | 0.1 × 0.1 × 0.6 × 0.6 | **0.0036** | 0.0036 ✓ |

### Parte 2 — Ejemplo de clase con traza

| Variable consulta | Evidencia | P(attend) | P(miss) | Esperado |
|------------------|-----------|-----------|---------|---------|
| Appointment | Rain=light, Maintenance=no | **0.8100** | **0.1900** | 0.81 / 0.19 ✓ |

### Parte 3 — Consultas adicionales

| Consulta | Evidencia | Resultado |
|----------|-----------|-----------|
| `P(Train)` | Rain=heavy | on_time=0.49, delayed=0.51 |
| `P(Appointment)` | (ninguna) | attend=0.8361, miss=0.1639 |
| `P(Rain)` | Appointment=miss | none=0.6065, light=0.2392, heavy=0.1544 |
| `P(Maintenance)` | Train=on_time | yes=0.3202, no=0.6798 |
| `P(Train)` | Rain=light, Maintenance=yes | on_time=0.60, delayed=0.40 |
| `P(Appointment)` | Rain=none | attend=0.8580, miss=0.1420 |

Todas las distribuciones suman exactamente 1.0, lo que confirma la corrección del algoritmo.

**Observaciones sobre las consultas inversas:**

- `P(Rain | Appointment=miss)`: dado que se falló la reunión, aumenta ligeramente la probabilidad de lluvia fuerte (0.154 vs. 0.10 a priori), lo que tiene sentido causalmente.
- `P(Maintenance | Train=on_time)`: si el tren llegó a tiempo, es más probable que no hubiera mantenimiento (0.68), ya que el mantenimiento tiende a retrasar los trenes.

---

## 7. Cómo ejecutar

```bash
# Posicionarse dentro del directorio del proyecto
cd ProyectoIA3

# Ejecutar el programa completo
python3 main.py
```

**Requisitos:** Python 3.x estándar (sin dependencias externas).

### Adaptar a otro dominio

Para cambiar el dominio de la red basta modificar los dos archivos de texto; el código Python no requiere ningún cambio:

1. Editar `network_structure.txt` con los nuevos arcos.
2. Editar `network_probabilities.txt` con los nuevos nodos, dominios y CPTs.
3. En `main.py`, actualizar las llamadas a `enumerate_ask()` con los nuevos nombres de variables.

### Opciones del motor de inferencia

```python
# Con traza detallada (muestra cada paso recursivo)
resultado = bn.enumerate_ask("NodoX", {"NodoY": "valor"}, trace=True)

# Sin traza (solo devuelve la distribución)
resultado = bn.enumerate_ask("NodoX", {"NodoY": "valor"}, trace=False)

# Mostrar resultado en formato compacto
bn.display_query_result("NodoX", evidencia, resultado)
```

El valor de retorno es un diccionario `{valor: probabilidad}`, por ejemplo:

```python
{"attend": 0.81, "miss": 0.19}
```

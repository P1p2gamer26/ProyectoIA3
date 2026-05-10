"""
Proyecto 3 - Motor de Inferencia por Enumeracion
Parte 1: Red Bayesiana - Estructura y Tablas de Probabilidad

Clases implementadas:
  CPT             - Tabla de Probabilidad Condicional
  Node            - Nodo (variable aleatoria) del grafo
  Arc             - Arco dirigido entre dos nodos
  BayesianNetwork - Grafo Dirigido Aciclico (DAG)
"""


# =============================================================================
# Clase CPT - Conditional Probability Table
# =============================================================================

class CPT:
    """
    Tabla de Probabilidad Condicional de un nodo dado sus padres.

    Almacena P(nodo=v | padres=combinacion) para cada combinacion de
    valores de los padres y cada valor v del nodo.

    La clave interna es una tupla con los valores de los padres (en el
    mismo orden en que se declararon en PARENTS); el valor asociado es
    un diccionario {valor_nodo: probabilidad}.
    """

    def __init__(self, node_values):
        """
        Parametros:
            node_values (list[str]): valores posibles del nodo,
                                     ej. ['on_time', 'delayed'].
        """
        self.node_values = node_values   # dominio del nodo
        self.entries = {}                # {tupla_padres: {valor: prob}}

    # ------------------------------------------------------------------
    def add_entry(self, parent_vals_tuple, probabilities):
        """
        Agrega una fila completa a la tabla.

        Parametros:
            parent_vals_tuple (tuple): valores de los padres para esta
                fila; tupla vacia si el nodo no tiene padres.
            probabilities (list[float]): probabilidades en el mismo
                orden que self.node_values.
        """
        if len(probabilities) != len(self.node_values):
            raise ValueError(
                f"Se esperaban {len(self.node_values)} probabilidades, "
                f"se recibieron {len(probabilities)}."
            )
        self.entries[parent_vals_tuple] = dict(zip(self.node_values, probabilities))

    # ------------------------------------------------------------------
    def get_probability(self, node_value, parent_vals_tuple=()):
        """
        Retorna P(nodo=node_value | padres=parent_vals_tuple).

        Parametros:
            node_value (str)        : valor del nodo consultado.
            parent_vals_tuple (tuple): valores actuales de los padres.

        Retorna:
            float: probabilidad buscada.
        """
        if parent_vals_tuple not in self.entries:
            raise KeyError(
                f"Combinacion de padres {parent_vals_tuple} no encontrada en la CPT."
            )
        if node_value not in self.entries[parent_vals_tuple]:
            raise KeyError(f"Valor '{node_value}' no encontrado en la CPT.")
        return self.entries[parent_vals_tuple][node_value]

    # ------------------------------------------------------------------
    def display(self, node_name, parent_names):
        """
        Imprime la tabla en formato texto alineado por columnas.

        Parametros:
            node_name (str)       : nombre del nodo dueno de esta CPT.
            parent_names (list[str]): nombres de los padres (puede ser []).
        """
        all_headers = parent_names + self.node_values
        col_w = max((len(h) for h in all_headers), default=8) + 2

        sep = "  " + "-" * (col_w * len(all_headers))
        print(f"\n  CPT de: {node_name}")
        print(sep)
        print("  " + "".join(h.ljust(col_w) for h in all_headers))
        print(sep)

        for parent_tuple, probs in self.entries.items():
            row = "".join(str(pv).ljust(col_w) for pv in parent_tuple)
            row += "".join(str(probs[v]).ljust(col_w) for v in self.node_values)
            print("  " + row)

        print(sep)


# =============================================================================
# Clase Arc - Arco dirigido
# =============================================================================

class Arc:
    """
    Arco dirigido (padre -> hijo) que representa dependencia causal
    entre dos variables de la Red Bayesiana.
    """

    def __init__(self, source, destination):
        """
        Parametros:
            source (Node)     : nodo origen  (padre / causa).
            destination (Node): nodo destino (hijo  / efecto).
        """
        self.source = source
        self.destination = destination

    def __repr__(self):
        return f"Arc({self.source.name} -> {self.destination.name})"


# =============================================================================
# Clase Node - Nodo de la red
# =============================================================================

class Node:
    """
    Variable aleatoria dentro de la Red Bayesiana.

    Atributos:
        name     (str)       : identificador unico del nodo.
        values   (list[str]) : dominio de la variable.
        parents  (list[Node]): nodos padre (causas directas).
        children (list[Node]): nodos hijo  (efectos directos).
        cpt      (CPT)       : tabla de probabilidad P(nodo | padres).
    """

    def __init__(self, name, values=None):
        """
        Parametros:
            name   (str)           : nombre del nodo.
            values (list[str], opt): valores posibles; se puede asignar
                                     despues al cargar las probabilidades.
        """
        self.name = name
        self.values = values or []   # dominio de la variable
        self.parents = []            # [Node] - padres en el grafo
        self.children = []           # [Node] - hijos en el grafo
        self.cpt = None              # CPT asignada al cargar probabilidades

    # ------------------------------------------------------------------
    def add_parent(self, parent_node):
        """Registra un nodo como padre (evita duplicados)."""
        if parent_node not in self.parents:
            self.parents.append(parent_node)

    def add_child(self, child_node):
        """Registra un nodo como hijo (evita duplicados)."""
        if child_node not in self.children:
            self.children.append(child_node)

    def set_cpt(self, cpt):
        """Asigna la tabla de probabilidad condicional al nodo."""
        self.cpt = cpt

    # ------------------------------------------------------------------
    def get_probability(self, value, parent_assignment=None):
        """
        Retorna P(nodo=value | asignacion de padres).

        Parametros:
            value             (str) : valor del nodo.
            parent_assignment (dict): {nombre_padre: valor_padre}.
                                      Vacio o None si no hay padres.
        Retorna:
            float: probabilidad.
        """
        if parent_assignment is None:
            parent_assignment = {}
        # Construye la clave respetando el orden de self.parents
        parent_tuple = tuple(parent_assignment.get(p.name, "") for p in self.parents)
        return self.cpt.get_probability(value, parent_tuple)

    # ------------------------------------------------------------------
    def is_root(self):
        """Retorna True si el nodo no tiene padres (es raiz del DAG)."""
        return len(self.parents) == 0

    def __repr__(self):
        return f"Node({self.name})"


# =============================================================================
# Clase BayesianNetwork - Grafo Dirigido Aciclico
# =============================================================================

class BayesianNetwork:
    """
    Red Bayesiana: grafo dirigido aciclico (DAG) que modela relaciones
    de dependencia probabilistica entre variables aleatorias.

    Responsabilidades:
      - Cargar la estructura del grafo desde un archivo de texto.
      - Cargar las CPTs de cada nodo desde un archivo de texto.
      - Mostrar la estructura y las tablas en formato legible.
    """

    def __init__(self):
        self.nodes = {}   # {nombre_str: Node}
        self.arcs = []    # [Arc]

    # =========================================================================
    # Carga de estructura
    # =========================================================================

    def load_structure(self, filepath):
        """
        Lee la topologia de la red desde un archivo de texto.

        Formato (una relacion por linea):
            padre,hijo
        Lineas en blanco y las que comienzan con '#' se ignoran.

        Ejemplo:
            Rain,Maintenance
            Rain,Train
            Maintenance,Train
            Train,Appointment

        Parametros:
            filepath (str): ruta al archivo de estructura.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) != 2:
                    raise ValueError(
                        f"Linea {lineno}: formato invalido '{line}'. "
                        f"Se esperaba 'padre,hijo'."
                    )

                parent_name, child_name = parts

                # Crea los nodos si aun no existen
                if parent_name not in self.nodes:
                    self.nodes[parent_name] = Node(parent_name)
                if child_name not in self.nodes:
                    self.nodes[child_name] = Node(child_name)

                parent_node = self.nodes[parent_name]
                child_node = self.nodes[child_name]

                # Establece la relacion padre <-> hijo
                child_node.add_parent(parent_node)
                parent_node.add_child(child_node)

                # Registra el arco
                self.arcs.append(Arc(parent_node, child_node))

        print(f"[INFO] Estructura cargada desde '{filepath}'")
        print(f"       Nodos: {list(self.nodes.keys())}")
        print(f"       Arcos: {len(self.arcs)}")

    # =========================================================================
    # Carga de probabilidades
    # =========================================================================

    def load_probabilities(self, filepath):
        """
        Lee las tablas de probabilidad condicional (CPT) desde un archivo.

        Formato de cada bloque:
            NODE <nombre_nodo>
            VALUES <val1> <val2> ...
            [PARENTS <padre1> <padre2> ...]   <- omitir si no tiene padres
            <val_p1> [val_p2 ...] <prob1> <prob2> ...
            ... (una fila por combinacion de valores de los padres)

        Nodo SIN padres - una sola fila con todas las probs:
            NODE Rain
            VALUES none light heavy
            0.7 0.2 0.1

        Nodo CON padres - una fila por combinacion:
            NODE Train
            VALUES on_time delayed
            PARENTS Rain Maintenance
            none  yes  0.8 0.2
            none  no   0.9 0.1

        Lineas en blanco y las que comienzan con '#' se ignoran.

        Parametros:
            filepath (str): ruta al archivo de probabilidades.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [
                l.strip() for l in f
                if l.strip() and not l.strip().startswith('#')
            ]

        i = 0
        while i < len(lines):
            if not lines[i].upper().startswith('NODE'):
                i += 1
                continue

            tokens = lines[i].split()
            if len(tokens) < 2:
                raise ValueError(f"Linea '{lines[i]}': falta el nombre del nodo.")
            node_name = tokens[1]

            if node_name not in self.nodes:
                raise ValueError(
                    f"Nodo '{node_name}' en el archivo de probabilidades "
                    f"no existe en la estructura de la red."
                )
            node = self.nodes[node_name]
            i += 1

            # Lee la linea VALUES
            if i >= len(lines) or not lines[i].upper().startswith('VALUES'):
                raise ValueError(
                    f"Se esperaba 'VALUES' despues de 'NODE {node_name}'."
                )
            node.values = lines[i].split()[1:]
            i += 1

            # Lee la linea PARENTS (opcional)
            declared_parents = []
            if i < len(lines) and lines[i].upper().startswith('PARENTS'):
                declared_parents = lines[i].split()[1:]
                i += 1

            # Construye la CPT leyendo las filas de datos
            cpt = CPT(node.values)
            num_parent_cols = len(declared_parents)

            while i < len(lines) and not lines[i].upper().startswith('NODE'):
                row_tokens = lines[i].split()
                expected = num_parent_cols + len(node.values)

                if len(row_tokens) != expected:
                    raise ValueError(
                        f"Fila mal formada en CPT de '{node_name}': '{lines[i]}'. "
                        f"Se esperaban {expected} tokens "
                        f"({num_parent_cols} de padres + {len(node.values)} probabilidades)."
                    )

                parent_vals = tuple(row_tokens[:num_parent_cols])
                probs = [float(t) for t in row_tokens[num_parent_cols:]]
                cpt.add_entry(parent_vals, probs)
                i += 1

            node.set_cpt(cpt)

        print(f"[INFO] Probabilidades cargadas desde '{filepath}'")

    # =========================================================================
    # Visualizacion de la estructura
    # =========================================================================

    def display_structure(self):
        """
        Imprime la estructura de la red en formato texto.

        Muestra:
          1. Arbol jerarquico recorrido desde cada nodo raiz.
          2. Resumen de predecesores y sucesores por nodo.
        """
        print("\n" + "=" * 60)
        print("ESTRUCTURA DE LA RED BAYESIANA")
        print("=" * 60)

        roots = self.get_roots()
        if not roots:
            print("[ADVERTENCIA] No se encontraron nodos raiz (posible ciclo).")
            return

        print(f"\nNodos raiz : {[r.name for r in roots]}")
        print(f"Total nodos: {len(self.nodes)}")
        print(f"Total arcos: {len(self.arcs)}")
        print("\nJerarquia (desde raices):")

        visited = set()
        for idx, root in enumerate(roots):
            self._print_tree(root, visited, prefix="", is_last=(idx == len(roots) - 1))

        print("\nResumen de dependencias por nodo:")
        print("-" * 50)
        for name, node in self.nodes.items():
            pred = ", ".join(p.name for p in node.parents) or "(ninguno)"
            succ = ", ".join(c.name for c in node.children) or "(ninguno)"
            print(f"  {name}")
            print(f"    Predecesores : {pred}")
            print(f"    Sucesores    : {succ}")

        print("=" * 60)

    def _print_tree(self, node, visited, prefix, is_last):
        """
        Recorrido DFS recursivo para imprimir el arbol con sangria.

        Parametros:
            node    (Node): nodo actual.
            visited (set) : nodos ya impresos (evita repeticion en DAGs).
            prefix  (str) : sangria acumulada.
            is_last (bool): si es el ultimo hermano en su nivel.
        """
        connector = "+-- "
        already = " (ya mostrado)" if node.name in visited else ""
        parent_tag = ""
        if node.parents:
            parent_tag = f"  [padres: {', '.join(p.name for p in node.parents)}]"

        print(f"{prefix}{connector}{node.name}{parent_tag}{already}")

        if node.name in visited:
            return
        visited.add(node.name)

        child_prefix = prefix + ("    " if is_last else "|   ")
        for i, child in enumerate(node.children):
            self._print_tree(
                child, visited, child_prefix,
                is_last=(i == len(node.children) - 1)
            )

    # =========================================================================
    # Visualizacion de tablas de probabilidad
    # =========================================================================

    def display_probabilities(self):
        """
        Imprime las CPTs de todos los nodos de la red en formato tabular.
        """
        print("\n" + "=" * 60)
        print("TABLAS DE PROBABILIDAD CONDICIONAL (CPT)")
        print("=" * 60)

        for name, node in self.nodes.items():
            if node.cpt is None:
                print(f"\n  [AVISO] Nodo '{name}' no tiene CPT cargada.")
                continue
            parent_names = [p.name for p in node.parents]
            node.cpt.display(name, parent_names)

        print("=" * 60)

    # =========================================================================
    # Utilidades
    # =========================================================================

    def get_roots(self):
        """Retorna la lista de nodos sin padres (raices del DAG)."""
        return [n for n in self.nodes.values() if n.is_root()]

    def get_node(self, name):
        """
        Obtiene un nodo por su nombre.

        Parametros:
            name (str): nombre del nodo.

        Retorna:
            Node: el nodo solicitado.
        """
        if name not in self.nodes:
            raise KeyError(f"Nodo '{name}' no encontrado en la red.")
        return self.nodes[name]

    def topological_sort(self):
        """
        Retorna los nodos en orden topologico (padres antes que hijos).
        Util para el motor de inferencia por enumeracion.

        Retorna:
            list[Node]: nodos ordenados topologicamente.
        """
        visited = set()
        order = []

        def dfs(node):
            if node.name in visited:
                return
            visited.add(node.name)
            for parent in node.parents:
                dfs(parent)
            order.append(node)

        for node in self.nodes.values():
            dfs(node)

        return order

"""
Proyecto 3 - Motor de Inferencia por Enumeracion
Parte 1: Demostracion de Red Bayesiana

Carga la red del ejemplo del Tren y la Reunion, luego muestra:
  1. La estructura de la red (recorrido desde raices).
  2. Las tablas de probabilidad condicional de cada nodo.
  3. Verificacion de probabilidades conjuntas con valores conocidos.
"""

import os
import sys
import io

# Fuerza UTF-8 en la salida para mostrar acentos correctamente en Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from bayesian_network import BayesianNetwork


STRUCTURE_FILE     = "network_structure.txt"
PROBABILITIES_FILE = "network_probabilities.txt"


def check_files(*paths):
    """Verifica que todos los archivos necesarios existan."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[ERROR] Archivo no encontrado: '{p}'")
        sys.exit(1)


def verify_joint_probabilities(bn):
    """
    Calcula dos probabilidades conjuntas para verificar la correcta
    carga de la estructura y las CPTs.

    Caso 1: lluvia ligera, sin mantenimiento, tren retrasado, falta reunion.
      P = P(light) * P(no|light) * P(delayed|light,no) * P(miss|delayed)
        = 0.2 * 0.8 * 0.3 * 0.4 = 0.0192

    Caso 2: lluvia fuerte, con mantenimiento, tren retrasado, asiste reunion.
      P = P(heavy) * P(yes|heavy) * P(delayed|heavy,yes) * P(attend|delayed)
        = 0.1 * 0.1 * 0.6 * 0.6 = 0.0036
    """
    print("\n" + "=" * 60)
    print("VERIFICACION DE PROBABILIDADES CONJUNTAS")
    print("=" * 60)

    rain  = bn.get_node("Rain")
    maint = bn.get_node("Maintenance")
    train = bn.get_node("Train")
    appt  = bn.get_node("Appointment")

    # --- Caso 1 ---
    print("\nCaso 1: Rain=light, Maintenance=no, Train=delayed, Appointment=miss")

    p1 = (rain.get_probability("light")
          * maint.get_probability("no",      {"Rain": "light"})
          * train.get_probability("delayed", {"Rain": "light", "Maintenance": "no"})
          * appt.get_probability("miss",     {"Train": "delayed"}))

    print(f"  P = P(light) * P(no|light) * P(delayed|light,no) * P(miss|delayed)")
    print(f"  P = {p1:.4f}  (esperado: 0.0192)")
    print(f"  Resultado: {'OK' if abs(p1 - 0.0192) < 1e-9 else 'ERROR'}")

    # --- Caso 2 ---
    print("\nCaso 2: Rain=heavy, Maintenance=yes, Train=delayed, Appointment=attend")

    p2 = (rain.get_probability("heavy")
          * maint.get_probability("yes",     {"Rain": "heavy"})
          * train.get_probability("delayed", {"Rain": "heavy", "Maintenance": "yes"})
          * appt.get_probability("attend",   {"Train": "delayed"}))

    print(f"  P = P(heavy) * P(yes|heavy) * P(delayed|heavy,yes) * P(attend|delayed)")
    print(f"  P = {p2:.4f}  (esperado: 0.0036)")
    print(f"  Resultado: {'OK' if abs(p2 - 0.0036) < 1e-9 else 'ERROR'}")

    print("=" * 60)


def main():
    print("=" * 60)
    print("RED BAYESIANA - Parte 1")
    print("Motor de Inferencia por Enumeracion")
    print("=" * 60)

    check_files(STRUCTURE_FILE, PROBABILITIES_FILE)

    bn = BayesianNetwork()

    # Carga la estructura del grafo
    bn.load_structure(STRUCTURE_FILE)

    # Muestra la estructura (recorrido desde raices)
    bn.display_structure()

    # Carga las tablas de probabilidad condicional
    bn.load_probabilities(PROBABILITIES_FILE)

    # Muestra las CPTs de todos los nodos
    bn.display_probabilities()

    # Verificacion con probabilidades conjuntas conocidas
    verify_joint_probabilities(bn)

    print("\n[INFO] Parte 1 completada correctamente.")


if __name__ == "__main__":
    main()

"""
Proyecto 3 - Motor de Inferencia por Enumeracion
Archivo principal de ejecucion y demostracion

Estructura del programa:
  Parte 1 - Red Bayesiana
    - Carga la estructura del grafo desde archivo.
    - Carga las tablas de probabilidad condicional desde archivo.
    - Muestra la estructura y las CPTs en formato texto.
    - Verifica dos probabilidades conjuntas conocidas del ejemplo de clase.

  Parte 2 - Motor de Inferencia por Enumeracion
    - Ejecuta el ejemplo de clase con traza detallada paso a paso.
    - Ejecuta consultas adicionales sobre la misma red para validar
      la genericidad del motor (incluyendo inferencia "hacia atras").

  Parte 3 - Validacion adicional
    - Prueba inferencias para todos los nodos de la red.
    - Verifica que las distribuciones resultantes sumen 1.0.

La red modelada es el ejemplo del Tren y la Reunion, con cuatro nodos:
    Rain -> Maintenance -> Train -> Appointment
             Rain ->-----------------^
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


# =============================================================================
# Utilidades
# =============================================================================

def check_files(*paths):
    """Verifica que todos los archivos necesarios existan antes de continuar."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[ERROR] Archivo no encontrado: '{p}'")
        sys.exit(1)


# =============================================================================
# Parte 1: Demostracion de la estructura y CPTs
# =============================================================================

def verify_joint_probabilities(bn):
    """
    Calcula dos probabilidades conjuntas para verificar la correcta
    carga de la estructura y las CPTs.

    Las probabilidades conjuntas se calculan usando la regla de la cadena:
        P(A, B, C, D) = P(A) * P(B|A) * P(C|A,B) * P(D|C)

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

    # --- Caso 1: del ejemplo de clase (diapositiva 24) ---
    print("\nCaso 1: Rain=light, Maintenance=no, Train=delayed, Appointment=miss")
    print("  (Ejemplo directo de la clase de Metodos Probabilisticos)")

    p1 = (rain.get_probability("light")
          * maint.get_probability("no",      {"Rain": "light"})
          * train.get_probability("delayed", {"Rain": "light", "Maintenance": "no"})
          * appt.get_probability("miss",     {"Train": "delayed"}))

    print(f"  P = P(light) * P(no|light) * P(delayed|light,no) * P(miss|delayed)")
    print(f"  P = 0.2 * 0.8 * 0.3 * 0.4 = {p1:.4f}  (esperado: 0.0192)")
    print(f"  Resultado: {'[OK]' if abs(p1 - 0.0192) < 1e-9 else '[ERROR]'}")

    # --- Caso 2: del ejemplo de clase (diapositiva 25) ---
    print("\nCaso 2: Rain=heavy, Maintenance=yes, Train=delayed, Appointment=attend")

    p2 = (rain.get_probability("heavy")
          * maint.get_probability("yes",     {"Rain": "heavy"})
          * train.get_probability("delayed", {"Rain": "heavy", "Maintenance": "yes"})
          * appt.get_probability("attend",   {"Train": "delayed"}))

    print(f"  P = P(heavy) * P(yes|heavy) * P(delayed|heavy,yes) * P(attend|delayed)")
    print(f"  P = 0.1 * 0.1 * 0.6 * 0.6 = {p2:.4f}  (esperado: 0.0036)")
    print(f"  Resultado: {'[OK]' if abs(p2 - 0.0036) < 1e-9 else '[ERROR]'}")

    print("=" * 60)


# =============================================================================
# Parte 2: Motor de Inferencia - Consultas con traza
# =============================================================================

def run_inference_with_trace(bn):
    """
    Ejecuta la consulta del ejemplo de clase con traza detallada paso a paso.

    La traza permite visualizar exactamente como el algoritmo ENUMERATION-ASK
    recorre las variables en orden topologico, distinguiendo entre:
      - Variables con valor conocido (evidencia): se usa directamente su P.
      - Variables ocultas: se marginaliza sumando sobre todos sus valores.

    Consulta del ejemplo (diapositiva 29 de la clase):
        P(Appointment | Rain=light, Maintenance=no)
    Resultado esperado:
        P(attend | light, no) = 0.81
        P(miss   | light, no) = 0.19
    """
    print("\n\n" + "#" * 60)
    print("# PARTE 2 - MOTOR DE INFERENCIA POR ENUMERACION")
    print("#" * 60)

    print("\n--- Consulta del ejemplo de clase (con traza detallada) ---")
    print("Objetivo: P(Appointment | Rain=light, Maintenance=no)")
    print("(Diapositiva 29 - Metodos Probabilisticos)\n")

    evidence = {"Rain": "light", "Maintenance": "no"}

    # trace=True activa la impresion de cada paso del algoritmo recursivo
    result = bn.enumerate_ask("Appointment", evidence, trace=True)

    # Verificamos que el resultado coincida con el esperado de la clase
    expected_attend = 0.81
    expected_miss   = 0.19
    ok_attend = abs(result.get("attend", 0) - expected_attend) < 0.001
    ok_miss   = abs(result.get("miss",   0) - expected_miss)   < 0.001

    print(f"\n  Verificacion vs. ejemplo de clase:")
    print(f"    P(attend) esperado: {expected_attend}  obtenido: {result.get('attend', 0):.4f}  {'[OK]' if ok_attend else '[ERROR]'}")
    print(f"    P(miss)   esperado: {expected_miss}  obtenido: {result.get('miss',   0):.4f}  {'[OK]' if ok_miss else '[ERROR]'}")


# =============================================================================
# Parte 3: Validacion con multiples consultas
# =============================================================================

def run_additional_queries(bn):
    """
    Ejecuta un conjunto de consultas adicionales para validar que el motor
    funciona correctamente en casos distintos al ejemplo de clase.

    Las consultas cubren:
      1. Inferencia directa (causa -> efecto): lluvia fuerte, que pasa con el tren.
      2. Inferencia sin evidencia: distribucion a priori de la reunion.
      3. Inferencia inversa (efecto -> causa): dado que falle, que tan probable
         es que haya llovido (el motor lo maneja igual que cualquier otra consulta).
      4. Inferencia intermedia: estado del tren dado solo la lluvia.
      5. Consulta sobre nodo con multiples padres: tren dado lluvia y mantenimiento.
    """
    print("\n\n" + "#" * 60)
    print("# PARTE 3 - VALIDACION CON MULTIPLES CONSULTAS")
    print("#" * 60)
    print("(Sin traza detallada para mayor claridad)\n")

    consultas = [
        # (descripcion, variable_consulta, evidencia)
        (
            "1. Consulta directa: Estado del tren con lluvia fuerte",
            "Train",
            {"Rain": "heavy"}
        ),
        (
            "2. Sin evidencia: Distribucion a priori de la reunion",
            "Appointment",
            {}
        ),
        (
            "3. Inferencia inversa: Lluvia probable dado que falle la reunion",
            "Rain",
            {"Appointment": "miss"}
        ),
        (
            "4. Inferencia inversa: Mantenimiento probable dado que el tren llego a tiempo",
            "Maintenance",
            {"Train": "on_time"}
        ),
        (
            "5. Consulta con padres multiples: Tren dado lluvia leve y mantenimiento",
            "Train",
            {"Rain": "light", "Maintenance": "yes"}
        ),
        (
            "6. Consulta encadenada: Reunion dado solo el tipo de lluvia",
            "Appointment",
            {"Rain": "none"}
        ),
    ]

    for descripcion, query_var, evidence in consultas:
        print(f"\n  {descripcion}")
        print(f"  " + "-" * 50)
        result = bn.enumerate_ask(query_var, evidence, trace=False)
        bn.display_query_result(query_var, evidence, result)

        # Verificamos que las probabilidades sumen 1 (condicion de coherencia)
        total = sum(result.values())
        print(f"    [Suma de probabilidades = {total:.6f}  "
              f"{'OK' if abs(total - 1.0) < 1e-9 else 'ERROR - no suma 1'}]")

    print("\n" + "=" * 60)
    print("Todas las consultas completadas.")
    print("=" * 60)


# =============================================================================
# Programa principal
# =============================================================================

def main():
    print("=" * 60)
    print("PROYECTO 3 - MOTOR DE INFERENCIA POR ENUMERACION")
    print("Red Bayesiana: Tren y Reunion")
    print("=" * 60)

    # Verificar existencia de archivos de datos
    check_files(STRUCTURE_FILE, PROBABILITIES_FILE)

    # --- Inicializacion de la red ---
    bn = BayesianNetwork()

    # Carga la topologia del grafo (arcos y nodos)
    bn.load_structure(STRUCTURE_FILE)

    # Muestra el grafo en formato texto (recorrido desde raices)
    bn.display_structure()

    # Carga las tablas de probabilidad condicional de cada nodo
    bn.load_probabilities(PROBABILITIES_FILE)

    # Muestra todas las CPTs en formato tabular
    bn.display_probabilities()

    print("\n" + "=" * 60)
    print("PARTE 1 - VERIFICACION DE PROBABILIDADES CONJUNTAS")
    print("=" * 60)

    # Verifica que la carga fue correcta usando probabilidades conocidas
    verify_joint_probabilities(bn)

    print("\n[INFO] Parte 1 completada correctamente.")

    # --- Parte 2: Inferencia con traza ---
    run_inference_with_trace(bn)

    print("\n[INFO] Parte 2 completada correctamente.")

    # --- Parte 3: Validacion adicional ---
    run_additional_queries(bn)

    print("\n[INFO] Demostracion completa finalizada.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Taller: Búsqueda en Grafos y Resolución de Laberintos
Algoritmos: DFS, BFS, Dijkstra, A*
"""

# ── Imports (todos al inicio) ─────────────────────────────────────────────────
import random
import numpy as np
import sys
import time
from collections import deque
import heapq

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ── Constantes ────────────────────────────────────────────────────────────────
CAMINO   = 0
PARED    = 1
ORIGEN   = 2
META     = 3
VISITADO = 4
SOLUCION = 5

# Paleta de colores: [Blanco, Negro, Azul, Rojo, Celeste, Amarillo]
colores_laberinto = ListedColormap(
    ['#FFFFFF', '#1A1A1A', '#007BFF', '#DC3545', '#AED6F1', '#F1C40F']
)

# ── Parámetros ────────────────────────────────────────────────────────────────
sys.setrecursionlimit(5000)
probabilidad_romper_pared = 0.05
ANCHO = 200
ALTO  = 200


# ── Generación del laberinto ──────────────────────────────────────────────────
def generar_laberinto(ancho, alto, probabilidad_romper):
    """
    Genera una matriz bidimensional que representa un laberinto aleatorio
    utilizando búsqueda en profundidad (DFS) con backtracking.

    Args:
        ancho (int): El número de columnas (se ajustará a impar).
        alto  (int): El número de filas  (se ajustará a impar).
        probabilidad_romper (float): probabilidad [0-1] de romper una pared.

    Returns:
        list[list[int]]: Matriz con valores CAMINO, PARED, ORIGEN y META.
    """
    if ancho % 2 == 0: ancho += 1
    if alto  % 2 == 0: alto  += 1

    # Inicializar lleno de paredes
    laberinto = [[PARED for _ in range(ancho)] for _ in range(alto)]

    def excavar(x, y):
        laberinto[y][x] = CAMINO
        direcciones = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        random.shuffle(direcciones)
        for dx, dy in direcciones:
            nx, ny = x + dx, y + dy
            if 1 <= nx < ancho - 1 and 1 <= ny < alto - 1 and laberinto[ny][nx] == PARED:
                laberinto[y + dy // 2][x + dx // 2] = CAMINO
                excavar(nx, ny)

    # 1. Generar laberinto base (perfecto)
    inicio_x = random.randrange(1, ancho, 2)
    inicio_y = random.randrange(1, alto,  2)
    excavar(inicio_x, inicio_y)

    # 2. Romper paredes aleatoriamente para crear ciclos (caminos múltiples)
    for y in range(1, alto - 1):
        for x in range(1, ancho - 1):
            if laberinto[y][x] == PARED:
                es_pared_horizontal = (laberinto[y][x-1] == CAMINO and laberinto[y][x+1] == CAMINO)
                es_pared_vertical   = (laberinto[y-1][x] == CAMINO and laberinto[y+1][x] == CAMINO)
                if (es_pared_horizontal or es_pared_vertical) and random.random() < probabilidad_romper:
                    laberinto[y][x] = CAMINO

    # 3. Establecer puntos de interés
    laberinto[1][0]               = ORIGEN
    laberinto[alto - 2][ancho - 1] = META

    return laberinto


# ── Visualización ─────────────────────────────────────────────────────────────
def dibujar_laberinto_grafico(laberinto):
    """Usa Matplotlib para renderizar la matriz del laberinto con colores y leyenda."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(laberinto, cmap=colores_laberinto, interpolation='nearest', vmin=0, vmax=5)
    ax.axis('off')
    plt.title("Laberinto", fontsize=18, fontweight='bold', pad=20)

    parches = [
        mpatches.Patch(color='#1A1A1A', label='Pared'),
        mpatches.Patch(color='#FFFFFF', label='Camino',   ec='#CCCCCC'),
        mpatches.Patch(color='#007BFF', label='Origen'),
        mpatches.Patch(color='#DC3545', label='Meta'),
        mpatches.Patch(color='#AED6F1', label='Explorado'),
        mpatches.Patch(color='#F1C40F', label='Solución'),
    ]
    ax.legend(handles=parches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              fancybox=True, shadow=True, ncol=3, fontsize=11)
    plt.tight_layout()
    plt.show()


def visualizar_resultado(laberinto, visitados, camino, titulo="Resultado", tiempo=None):
    """
    Renderiza el laberinto mostrando los nodos visitados y el camino solución.
    El título incluye las métricas (Visitados, Pasos, Tiempo) tal como
    se muestra en el enunciado del taller.
    """
    # Copiar la matriz para no modificar el laberinto original
    mapa = [fila[:] for fila in laberinto]

    # Marcar nodos visitados (sin sobreescribir ORIGEN ni META)
    for (i, j) in visitados:
        if mapa[i][j] not in (ORIGEN, META):
            mapa[i][j] = VISITADO

    # Marcar camino solución encima de los visitados
    for (i, j) in camino:
        if mapa[i][j] not in (ORIGEN, META):
            mapa[i][j] = SOLUCION

    mapa_np = np.array(mapa)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(mapa_np, cmap=colores_laberinto, interpolation='nearest', vmin=0, vmax=5)
    ax.axis('off')

    # Título con métricas (igual al ejemplo del PDF)
    tiempo_str = f"\n{tiempo:.5f}s" if tiempo is not None else ""
    titulo_completo = (
        f"{titulo}\n"
        f"Visitados: {len(visitados)}\n"
        f"Pasos: {len(camino)}"
        f"{tiempo_str}"
    )
    plt.title(titulo_completo, fontsize=14, fontweight='bold', pad=10)

    parches = [
        mpatches.Patch(color='#1A1A1A', label='Pared'),
        mpatches.Patch(color='#FFFFFF', label='Camino',   ec='#CCCCCC'),
        mpatches.Patch(color='#007BFF', label='Origen'),
        mpatches.Patch(color='#DC3545', label='Meta'),
        mpatches.Patch(color='#AED6F1', label='Explorado'),
        mpatches.Patch(color='#F1C40F', label='Solución'),
    ]
    ax.legend(handles=parches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              fancybox=True, shadow=True, ncol=3, fontsize=11)
    plt.tight_layout()
    plt.show()


# ── Utilidades ────────────────────────────────────────────────────────────────
def obtener_posicion(laberinto, valor):
    """Encuentra las coordenadas (f, c) de un valor específico en la matriz."""
    for f, fila in enumerate(laberinto):
        for c, celda in enumerate(fila):
            if celda == valor:
                return (f, c)
    return None


def construir_grafo(laberinto):
    """
    Convierte la matriz del laberinto en un diccionario de adyacencia.
    Solo conecta celdas que sean CAMINO, ORIGEN o META.
    """
    filas    = len(laberinto)
    columnas = len(laberinto[0])
    laberinto_grafos = {}
    transitables = {CAMINO, ORIGEN, META}

    for f in range(filas):
        for c in range(columnas):
            if laberinto[f][c] in transitables:
                nodo_actual = (f, c)
                laberinto_grafos[nodo_actual] = []

                for df, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nf, nc = f + df, c + dc
                    if 0 <= nf < filas and 0 <= nc < columnas:
                        if laberinto[nf][nc] in transitables:
                            laberinto_grafos[nodo_actual].append((nf, nc))

    return laberinto_grafos


# ── Algoritmos de búsqueda ────────────────────────────────────────────────────
def dfs(grafo, inicio, meta):
    """
    Búsqueda en Profundidad (DFS).
    Mecánica: Pila (Stack). No garantiza el camino más corto.
    """
    inicio_tiempo = time.time()
    stack    = [(inicio, [inicio])]
    visitados = set()

    while stack:
        nodo, camino = stack.pop()

        if nodo in visitados:
            continue
        visitados.add(nodo)

        if nodo == meta:
            return camino, visitados, (time.time() - inicio_tiempo)

        for vecino in grafo[nodo]:
            if vecino not in visitados:
                stack.append((vecino, camino + [vecino]))

    return None, visitados, None


def bfs(grafo, inicio, meta):
    """
    Búsqueda en Anchura (BFS).
    Mecánica: Cola (Queue). Garantiza el camino más corto en grafos sin pesos.
    """
    inicio_tiempo = time.time()
    queue     = deque([(inicio, [inicio])])
    visitados = set()

    while queue:
        nodo, camino = queue.popleft()

        if nodo in visitados:
            continue
        visitados.add(nodo)

        if nodo == meta:
            return camino, visitados, (time.time() - inicio_tiempo)

        for vecino in grafo[nodo]:
            if vecino not in visitados:
                queue.append((vecino, camino + [vecino]))

    return None, visitados, None


def dijkstra(grafo, inicio, meta):
    """
    Algoritmo de Dijkstra.
    Mecánica: Cola de prioridad (min-heap). Cuenta pasos como costo de cada arista.
    """
    inicio_tiempo = time.time()
    heap      = [(0, inicio, [inicio])]
    visitados = set()

    while heap:
        costo, nodo, camino = heapq.heappop(heap)

        if nodo in visitados:
            continue
        visitados.add(nodo)

        if nodo == meta:
            return camino, visitados, (time.time() - inicio_tiempo)

        for vecino in grafo[nodo]:
            if vecino not in visitados:
                heapq.heappush(heap, (costo + 1, vecino, camino + [vecino]))

    return None, visitados, None


def heuristica(a, b):
    """Distancia Manhattan entre dos coordenadas (fila, columna)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grafo, inicio, meta):
    """
    Algoritmo A* (A-Star).
    Mecánica: Cola de prioridad con f(n) = g(n) + h(n).
    Heurística: Distancia Manhattan al nodo meta.
    """
    inicio_tiempo = time.time()
    heap      = [(0, inicio, [inicio])]
    visitados = set()

    while heap:
        costo, nodo, camino = heapq.heappop(heap)

        if nodo in visitados:
            continue
        visitados.add(nodo)

        if nodo == meta:
            return camino, visitados, (time.time() - inicio_tiempo)

        for vecino in grafo[nodo]:
            if vecino not in visitados:
                prioridad = costo + 1 + heuristica(vecino, meta)
                heapq.heappush(heap, (prioridad, vecino, camino + [vecino]))

    return None, visitados, None


# ── Ejecución principal ───────────────────────────────────────────────────────
print(f"Generando y dibujando laberinto de {ANCHO}x{ALTO}...")
laberinto = generar_laberinto(ANCHO, ALTO, probabilidad_romper_pared)
dibujar_laberinto_grafico(laberinto)

# Construir grafo de adyacencia
grafo = construir_grafo(laberinto)
print(f"Grafo generado con {len(grafo)} nodos transitables.")

# Obtener posiciones de inicio y meta
inicio = obtener_posicion(laberinto, ORIGEN)
meta   = obtener_posicion(laberinto, META)
print(f"Inicio: {inicio}  |  Meta: {meta}")

# Ejecutar los cuatro algoritmos
camino_dfs,      visitados_dfs,      tiempo_dfs      = dfs(grafo, inicio, meta)
camino_bfs,      visitados_bfs,      tiempo_bfs      = bfs(grafo, inicio, meta)
camino_dijkstra, visitados_dijkstra, tiempo_dijkstra = dijkstra(grafo, inicio, meta)
camino_astar,    visitados_astar,    tiempo_astar    = a_star(grafo, inicio, meta)

# Tabla comparativa en consola
print("\n{:<12} {:>12} {:>18} {:>22}".format(
    "Algoritmo", "Tiempo (s)", "Nodos Visitados", "Longitud Camino"))
print("-" * 68)
for nombre, tiempo, visitados, camino in [
    ("DFS",      tiempo_dfs,      visitados_dfs,      camino_dfs),
    ("BFS",      tiempo_bfs,      visitados_bfs,      camino_bfs),
    ("Dijkstra", tiempo_dijkstra, visitados_dijkstra, camino_dijkstra),
    ("A*",       tiempo_astar,    visitados_astar,    camino_astar),
]:
    print("{:<12} {:>12.5f} {:>18} {:>22}".format(
        nombre, tiempo, len(visitados), len(camino)))

# Visualizar resultados con métricas en el título
visualizar_resultado(laberinto, visitados_dfs,      camino_dfs,      "DFS",      tiempo_dfs)
visualizar_resultado(laberinto, visitados_bfs,      camino_bfs,      "BFS",      tiempo_bfs)
visualizar_resultado(laberinto, visitados_dijkstra, camino_dijkstra, "Dijkstra", tiempo_dijkstra)
visualizar_resultado(laberinto, visitados_astar,    camino_astar,    "A*",       tiempo_astar)
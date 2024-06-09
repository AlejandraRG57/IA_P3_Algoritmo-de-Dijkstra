#Alejandra Rodriguez Guevara 21310127 6E1

#El algoritmo de Dijkstra es un método utilizado en teoría de grafos para determinar la ruta 
# o el camino más corto entre los nodos de un grafo.

#En el desarrollo de videojuegos, Dijkstra se puede utilizar para la inteligencia artificial de NPCs, 
# permitiéndoles moverse de manera eficiente en el entorno del juego.

import heapq #Importamos la biblioteca heapq para trabajar con colas de prioridad.
import matplotlib.pyplot as plt #Importamos pyplot de matplotlib para generar gráficos.
import networkx as nx #Importamos networkx para crear y manipular grafos.

class Graph:
    def __init__(self):
        self.nodes = set() #Inicializamos un conjunto vacío para almacenar los nodos.
        self.edges = {} #Inicializamos un diccionario vacío para almacenar las aristas y sus pesos.
    
    def add_node(self, value):
        self.nodes.add(value) #Agregamos el nodo al conjunto de nodos.
        if value not in self.edges:
            self.edges[value] = {} #Si el nodo no está en el diccionario de aristas, creamos una entrada para él.
    
    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node][to_node] = weight #Agregamos una arista del nodo origen al nodo destino con un peso.
        self.edges[to_node][from_node] = weight #Agregamos una arista del nodo destino al nodo origen con el mismo peso.
    
    def dijkstra(self, start, end=None):
        distances = {node: float('infinity') for node in self.nodes} #Inicializamos todas las distancias a infinito.
        distances[start] = 0 #La distancia al nodo inicial es 0.
        priority_queue = [(0, start)] #Creamos una cola de prioridad con el nodo inicial.
        previous_nodes = {} #Inicializamos un diccionario para los nodos previos en el camino más corto.
        all_paths = {} #Inicializamos un diccionario para almacenar todos los caminos intentados.
        
        while priority_queue: #Mientras la cola de prioridad no esté vacía.
            current_distance, current_node = heapq.heappop(priority_queue) #Obtenemos el nodo con la menor distancia.
            path = all_paths.get(current_node, []) #Obtenemos el camino actual.
            
            if current_node == end: #Si el nodo actual es el destino.
                path = [] #Inicializamos el camino.
                while current_node in previous_nodes: #Reconstruimos el camino desde el nodo destino al inicio.
                    path.insert(0, current_node) #Insertamos el nodo al inicio del camino.
                    current_node = previous_nodes[current_node] #Avanzamos al nodo previo.
                path.insert(0, start) #Insertamos el nodo inicial al inicio del camino.
                return current_distance, path, all_paths #Retornamos la distancia, el camino más corto y todos los caminos intentados.
            
            for neighbor, weight in self.edges[current_node].items(): #Para cada vecino del nodo actual.
                distance = current_distance + weight #Calculamos la nueva distancia.
                
                if distance < distances[neighbor]: #Si la nueva distancia es menor.
                    distances[neighbor] = distance #Actualizamos la distancia.
                    heapq.heappush(priority_queue, (distance, neighbor)) #Agregamos el vecino a la cola de prioridad.
                    previous_nodes[neighbor] = current_node #Actualizamos el nodo previo.
                    
                    #Agregamos el camino al vecino a la lista de caminos intentados.
                    all_paths[neighbor] = path + [neighbor] if current_node in all_paths else [start, neighbor]
        
        return float('infinity'), [], {} #Retornamos infinito si el nodo destino no es alcanzable.
    
    def draw(self, shortest_path=None, all_paths=None):
        G = nx.Graph() #Creamos un grafo vacío.
        
        #Definimos las posiciones de los nodos en el gráfico.
        positions = {
            'A': (5, 7), 'B': (7, 7.3), 'C': (8, 7), 'D': (3, 5), 
            'E': (5, 5), 'F': (7, 5), 'G': (9, 5), 'H': (2.75, 4),
            'I': (2.5, 3), 'J': (4, 3), 'K': (7, 3), 'L': (9, 3),
            'M': (5, 0.8), 'N': (7, 0.75), 'O': (8, 1)
        }
        
        #Agregamos los nodos al grafo.
        for node in self.nodes:
            G.add_node(node)
        
        #Agregamos las aristas al grafo con sus respectivos pesos.
        for node, edges in self.edges.items():
            for neighbor, weight in edges.items():
                G.add_edge(node, neighbor, weight=weight)
        
        #Dibujamos el grafo ponderado.
        plt.figure(figsize=(7, 7))
        plt.title('Mapa Videojuego')
        nx.draw(G, pos=positions, with_labels=True, node_color='skyblue', node_size=1500, font_size=10, font_weight='bold')
        
        #Agregamos etiquetas de peso a las aristas.
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels)
        
        #Dibujamos todos los caminos intentados si están especificados.
        if all_paths:
            for path in all_paths.values():
                edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(G, pos=positions, edgelist=edges, edge_color='gray', width=1, alpha=0.5)
        
        #Dibujamos el camino más corto si está especificado.
        if shortest_path:
            edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
            nx.draw_networkx_edges(G, pos=positions, edgelist=edges, edge_color='r', width=2)
        
        plt.title('Grafo ponderado del mapa del juego')
        plt.show()

game_map = Graph()

#Agregamos nodos (posiciones en el mapa)
game_map.add_node('A')
game_map.add_node('B')
game_map.add_node('C')
game_map.add_node('D')
game_map.add_node('E')
game_map.add_node('F')
game_map.add_node('G')
game_map.add_node('H')
game_map.add_node('I')
game_map.add_node('J')
game_map.add_node('K')
game_map.add_node('L')
game_map.add_node('M')
game_map.add_node('N')
game_map.add_node('O')

#Agregamos conexiones (caminos) entre nodos
game_map.add_edge('A', 'B', 2)
game_map.add_edge('A', 'D', 3) 
game_map.add_edge('B', 'A', 2) 
game_map.add_edge('B', 'C', 2) 
game_map.add_edge('B', 'F', 3)
game_map.add_edge('C', 'B', 2)
game_map.add_edge('C', 'G', 2)
game_map.add_edge('D', 'A', 3)
game_map.add_edge('D', 'E', 2)
game_map.add_edge('D', 'H', 1)
game_map.add_edge('E', 'D', 2)
game_map.add_edge('F', 'B', 3)
game_map.add_edge('F', 'G', 2)
game_map.add_edge('F', 'K', 4)
game_map.add_edge('G', 'C', 2)
game_map.add_edge('G', 'F', 2)
game_map.add_edge('G', 'L', 3)
game_map.add_edge('H', 'D', 1)
game_map.add_edge('H', 'I', 2)
game_map.add_edge('I', 'H', 2)
game_map.add_edge('I', 'J', 2)
game_map.add_edge('I', 'M', 5)
game_map.add_edge('J', 'I', 2)
game_map.add_edge('J', 'K', 3)
game_map.add_edge('K', 'F', 4)
game_map.add_edge('K', 'J', 3)
game_map.add_edge('K', 'M', 3)
game_map.add_edge('K', 'N', 3)
game_map.add_edge('L', 'G', 3)
game_map.add_edge('L', 'O', 2)
game_map.add_edge('M', 'I', 5)
game_map.add_edge('M', 'K', 3)
game_map.add_edge('M', 'N', 2)
game_map.add_edge('N', 'M', 2)
game_map.add_edge('N', 'O', 2)
game_map.add_edge('N', 'K', 3)
game_map.add_edge('O', 'N', 2)
game_map.add_edge('O', 'L', 2)

#Calculamos el camino más corto desde un nodo inicial hasta un nodo meta.
start_position = 'A'
end_position = 'M'
shortest_distance, shortest_path, all_paths = game_map.dijkstra(start_position, end_position)
print("\nDistancia mínima desde", start_position, "hasta", end_position, ":", shortest_distance, " km")
print("Camino tomado:", shortest_path)

#Dibujamos el grafo ponderado con los caminos intentados y el camino más corto resaltado.
game_map.draw(shortest_path, all_paths)

#Mostramos todos los caminos intentados en la consola.
print("\nTodos los caminos intentados:")
for node, path in all_paths.items():
    print("Desde", start_position, "a", node, ":", path)
import pygame
import numpy as np

GRID_SIZE = 18
CELL_SIZE = 40 
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
OUTPUTS = 10
MODEL_GRID_SIZE = 28

class Neuron:
    def __init__(self):
        self.connections = []
        self.bias = 0.0
        self.a = 0.0
        self.z = 0.0

    def init(self, prev_neurons):
        self.connections = [(0.0, neuron) for neuron in prev_neurons]

    def computeVal(self, clamp):
        self.z = self.bias
        for weight, prev_neuron in self.connections:
            self.z += weight * prev_neuron.a
        self.a = max(0.0, self.z) if clamp else self.z

class Layer:
    def __init__(self):
        self.neurons = []

    def init(self, n, prev_neurons):
        self.neurons = [Neuron() for _ in range(n)]
        for neuron in self.neurons:
            neuron.init(prev_neurons)
            
def softmax(z_values):
    z_values = np.array(z_values)
    max_z = np.max(z_values)
    exp_z = np.exp(z_values - max_z)
    return exp_z / np.sum(exp_z)

layers = []
LAYERS = 0
LAYERSIZE = 0
brush = []

def initLayers():
    global layers, LAYERS, LAYERSIZE
    with open("model.txt", "r") as f:
        lines = f.readlines()
    first_line = lines[0].strip().split()
    LAYERS = int(first_line[0])
    LAYERSIZE = int(first_line[1])
    values = []
    for line in lines[1:]:
        values.extend([float(x) for x in line.strip().split()])

    layers = [Layer() for _ in range(LAYERS + 2)]
    layers[0].init(MODEL_GRID_SIZE * MODEL_GRID_SIZE, [])
    for i in range(1, LAYERS + 1):
        layers[i].init(LAYERSIZE, layers[i-1].neurons)
    layers[LAYERS + 1].init(OUTPUTS, layers[LAYERS].neurons)

    idx = 0
    for l in range(1, LAYERS + 2):
        current_size = OUTPUTS if l == LAYERS + 1 else LAYERSIZE
        prev_size = MODEL_GRID_SIZE * MODEL_GRID_SIZE if l == 1 else LAYERSIZE
        for n in range(current_size):
            for p in range(prev_size):
                weight = values[idx]
                prev_neuron = layers[l-1].neurons[p]
                layers[l].neurons[n].connections[p] = (weight, prev_neuron)
                idx += 1
            layers[l].neurons[n].bias = values[idx]
            idx += 1

screen = None
grid = None
font = None

def setup():
    global screen, grid, font, brush
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    font = pygame.font.SysFont(None, 24)
    
    brush = [
        [0.2, 0.4, 0.2],
        [0.4, 1.0, 0.4],
        [0.2, 0.4, 0.2]
    ]

def apply_brush(grid, grid_x, grid_y):
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            x = grid_x + dx
            y = grid_y + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                brush_value = brush[dy + 1][dx + 1]
                grid[y][x] = max(grid[y][x], brush_value)

def pad_grid(grid):
    padded = [[0.0 for _ in range(MODEL_GRID_SIZE)] for _ in range(MODEL_GRID_SIZE)]
    offset = (MODEL_GRID_SIZE - GRID_SIZE) // 2
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            padded[y + offset][x + offset] = grid[y][x]
    return padded

def draw_grid(predictions):
    screen.fill(BLACK)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            value = grid[y][x]
            color = (int(255 * value), int(255 * value), int(255 * value))
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    if predictions is not None:
        max_idx = np.argmax(predictions)
        text = "Predictions:"
        color = WHITE
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (30, 30))
        for i in range(OUTPUTS):
            text = f"{i}: {predictions[i] * 100:.1f}%"
            color = GREEN if i == max_idx else WHITE
            text_surface = font.render(text, True, color)
            screen.blit(text_surface, (30, 60 + i * 30))
    text = "Press c to clear the grid"
    color = WHITE
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (30, HEIGHT - 60))
    pygame.display.flip()

def update_loop():
    global grid
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_x, mouse_y = event.pos
                if 0 <= mouse_x < WIDTH and 0 <= mouse_y < HEIGHT:
                    grid_x = mouse_x // CELL_SIZE
                    grid_y = mouse_y // CELL_SIZE
                    apply_brush(grid, grid_x, grid_y)
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:
                mouse_x, mouse_y = event.pos
                if 0 <= mouse_x < WIDTH and 0 <= mouse_y < HEIGHT:
                    grid_x = mouse_x // CELL_SIZE
                    grid_y = mouse_y // CELL_SIZE
                    apply_brush(grid, grid_x, grid_y)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


    padded_grid = pad_grid(grid)
    input_data = [padded_grid[y][x] for y in range(MODEL_GRID_SIZE) for x in range(MODEL_GRID_SIZE)]
    for i in range(MODEL_GRID_SIZE * MODEL_GRID_SIZE):
        layers[0].neurons[i].a = input_data[i]
    for i in range(1, LAYERS + 1):
        for neuron in layers[i].neurons:
            neuron.computeVal(True)
    for neuron in layers[LAYERS + 1].neurons:
        neuron.computeVal(False)
    z_output = [neuron.z for neuron in layers[LAYERS + 1].neurons]
    predictions = softmax(z_output)
    draw_grid(predictions)
    return True

if __name__ == "__main__":
    initLayers()
    setup()
    running = True
    while running:
        running = update_loop()
import pygame as pg
import numpy as np

NUMTEST = 40000
OUTPUTS = 10
RES = 28
LAYERS = 2
LAYERSIZE = 50

class Neuron:
    def __init__(self):
        self.connections = []
        self.bias = 0.0
        self.a = 0.0
        self.z = 0.0

    def init(self, prevN):
        self.connections = [(0.0, None)] * prevN

    @staticmethod
    def ReLU(x):
        return max(0.0, x)

    def computeVal(self, clamp):
        self.z = self.bias
        for weight, neuron in self.connections:
            self.z += weight * neuron.a
        self.a = self.ReLU(self.z) if clamp else self.z

class Layer:
    def __init__(self):
        self.neurons = []

    def init(self, n, prevN):
        self.neurons = [Neuron() for _ in range(n)]
        for neuron in self.neurons:
            neuron.init(prevN)

layers = []

def softmax():
    res = [0.0] * OUTPUTS
    max_val = max(neuron.z for neuron in layers[LAYERS + 1].neurons)
    tot = 0.0
    for i in range(OUTPUTS):
        res[i] = np.exp(layers[LAYERS + 1].neurons[i].z - max_val)
        tot += res[i]
    for i in range(OUTPUTS):
        res[i] /= tot
    return res

def initLayers():
    print("Creating the neural net...")
    try:
        with open("model.txt", "r") as cin2:
            global LAYERS, LAYERSIZE
            LAYERS, LAYERSIZE = map(int, cin2.readline().split())
            layers.extend([Layer() for _ in range(LAYERS + 2)])

            prev = RES * RES
            layers[0].init(prev, 0)

            for i in range(1, LAYERS + 1):
                layers[i].init(LAYERSIZE, prev)
                for neuron in layers[i].neurons:
                    for j in range(prev):
                        neuron.connections[j] = (0.0, layers[i - 1].neurons[j])
                prev = LAYERSIZE

            layers[LAYERS + 1].init(OUTPUTS, prev)
            for neuron in layers[LAYERS + 1].neurons:
                for j in range(prev):
                    neuron.connections[j] = (0.0, layers[LAYERS].neurons[j])

            for l in range(1, LAYERS + 2):
                currentSize = OUTPUTS if l == LAYERS + 1 else LAYERSIZE
                prevSize = RES * RES if l == 1 else LAYERSIZE

                for n in range(currentSize):
                    for p in range(prevSize):
                        try:
                            layers[l].neurons[n].connections[p] = (float(cin2.readline().strip()), layers[l - 1].neurons[p])
                        except ValueError:
                            print(f"Error reading weights at layer {l}, neuron {n}, connection {p}")
                            exit(1)

                    try:
                        layers[l].neurons[n].bias = float(cin2.readline().strip())
                    except ValueError:
                        print(f"Error reading bias at layer {l}, neuron {n}")
                        exit(1)

            dummy = cin2.readline()
            if dummy:
                print("Warning: Extra data in model file!")

            print("Network imported successfully.")
    except FileNotFoundError:
        print("Error: Could not open model file!")
        exit(1)

def getPreds():
    pixels = np.fliplr(grid)
    pixels = np.rot90(pixels, k=1)
    pixels = np.array(pixels, dtype=float).flatten()
    for i in range(RES * RES):
        layers[0].neurons[i].a = pixels[i]

    for i in range(1, LAYERS + 1):
        for j in range(LAYERSIZE):
            layers[i].neurons[j].computeVal(True)

    for i in range(OUTPUTS):
        layers[LAYERS + 1].neurons[i].computeVal(False)

    return softmax()

screen = None
grid = None
font = None
GRID_SIZE = 28
CELL_SIZE = 30
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
OUTPUTS = 10

def setup():
    global screen, grid, font, brush
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    font = pg.font.SysFont(None, 24)
    
    brush = [
        [0.1, 0.2, 0.3, 0.2, 0.1],
        [0.2, 0.6, 0.8, 0.6, 0.2],
        [0.3, 0.8, 1.0, 0.8, 0.3],
        [0.2, 0.6, 0.8, 0.6, 0.2],
        [0.1, 0.2, 0.3, 0.2, 0.1]
    ]

def apply_brush(grid, grid_x, grid_y):
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x = grid_x + dx
            y = grid_y + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                brush_value = brush[dy + 2][dx + 2]
                grid[y][x] = max(grid[y][x], brush_value)

def draw_grid():
    screen.fill(BLACK)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            value = grid[y][x]
            color = (int(255 * value), int(255 * value), int(255 * value))
            pg.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    text = "Press c to clear the grid"
    color = WHITE
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (30, HEIGHT - 60))
    preds = getPreds()
    for i in range(OUTPUTS):
        label = font.render(f"{i}: {preds[i]:.2f}", True, WHITE if preds[i] != max(preds) else GREEN)
        screen.blit(label, (10, 10 + i * 20))
        pg.draw.rect(screen, WHITE if preds[i] != max(preds) else GREEN, (70, 10 + i * 20, preds[i] * 50, 13))
    pg.display.flip()

def update_loop():
    global grid
    for event in pg.event.get():
        if event.type == pg.QUIT:
            return False
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_x, mouse_y = event.pos
                if 0 <= mouse_x < WIDTH and 0 <= mouse_y < HEIGHT:
                    grid_x = mouse_x // CELL_SIZE
                    grid_y = mouse_y // CELL_SIZE
                    apply_brush(grid, grid_x, grid_y)
        elif event.type == pg.MOUSEMOTION:
            if event.buttons[0]:
                mouse_x, mouse_y = event.pos
                if 0 <= mouse_x < WIDTH and 0 <= mouse_y < HEIGHT:
                    grid_x = mouse_x // CELL_SIZE
                    grid_y = mouse_y // CELL_SIZE
                    apply_brush(grid, grid_x, grid_y)
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_c:
                grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    draw_grid()
    return True


initLayers()
setup()
running = True
while running:
    running = update_loop()
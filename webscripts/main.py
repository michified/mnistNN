import asyncio
import math
from pathlib import Path

import pygame as pg


OUTPUTS = 10
RES = 28
LAYERS = 2
LAYERSIZE = 50
MODEL_LOCATIONS = (Path("model.txt"), Path(__file__).with_name("model.txt"))


class Neuron:
	def __init__(self):
		self.connections = []
		self.bias = 0.0
		self.a = 0.0
		self.z = 0.0

	def init(self, previous_count):
		self.connections = [(0.0, None)] * previous_count

	def compute_value(self, apply_activation):
		self.z = self.bias
		for weight, neuron in self.connections:
			self.z += weight * neuron.a
		self.a = max(0.0, self.z) if apply_activation else self.z


class Layer:
	def __init__(self):
		self.neurons = []

	def init(self, neuron_count, previous_count):
		self.neurons = [Neuron() for _ in range(neuron_count)]
		for neuron in self.neurons:
			neuron.init(previous_count)


layers = []


def softmax():
	logits = [neuron.z for neuron in layers[LAYERS + 1].neurons]
	maximum = max(logits)
	values = [math.exp(logit - maximum) for logit in logits]
	total = sum(values)
	return [value / total for value in values]


def open_model():
	for location in MODEL_LOCATIONS:
		try:
			return location.open("r")
		except FileNotFoundError:
			continue
	raise FileNotFoundError("model.txt was not included in the pygbag package")


def init_layers():
	global LAYERS, LAYERSIZE
	print("Creating the neural net...")
	layers.clear()
	try:
		with open_model() as model:
			header = model.readline().split()
			if len(header) != 2:
				raise ValueError("invalid model header")
			LAYERS, LAYERSIZE = map(int, header)
			layers.extend(Layer() for _ in range(LAYERS + 2))

			previous_count = RES * RES
			layers[0].init(previous_count, 0)
			for layer_index in range(1, LAYERS + 1):
				layers[layer_index].init(LAYERSIZE, previous_count)
				for neuron in layers[layer_index].neurons:
					neuron.connections = [
						(0.0, layers[layer_index - 1].neurons[index])
						for index in range(previous_count)
					]
				previous_count = LAYERSIZE

			layers[LAYERS + 1].init(OUTPUTS, previous_count)
			for neuron in layers[LAYERS + 1].neurons:
				neuron.connections = [
					(0.0, layers[LAYERS].neurons[index])
					for index in range(previous_count)
				]

			for layer_index in range(1, LAYERS + 2):
				current_count = OUTPUTS if layer_index == LAYERS + 1 else LAYERSIZE
				previous_count = RES * RES if layer_index == 1 else LAYERSIZE
				for neuron_index in range(current_count):
					neuron = layers[layer_index].neurons[neuron_index]
					for connection_index in range(previous_count):
						value = model.readline().strip()
						neuron.connections[connection_index] = (
							float(value),
							neuron.connections[connection_index][1],
						)
					bias = model.readline().strip()
					neuron.bias = float(bias)
	except (FileNotFoundError, ValueError) as error:
		raise RuntimeError(f"Could not load model.txt: {error}") from error
	print("Network imported successfully.")


def get_predictions(grid):
	pixels = [row[:] for row in grid]
	pixels = [row[::-1] for row in pixels]
	pixels = [list(column) for column in zip(*pixels)][::-1]
	pixels = [value for row in pixels for value in row]
	for index, value in enumerate(pixels):
		layers[0].neurons[index].a = value

	for layer_index in range(1, LAYERS + 1):
		for neuron in layers[layer_index].neurons:
			neuron.compute_value(True)
	for neuron in layers[LAYERS + 1].neurons:
		neuron.compute_value(False)
	return softmax()


GRID_SIZE = 28
CELL_SIZE = 30
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def apply_brush(grid, grid_x, grid_y):
	brush = (
		(0.1, 0.2, 0.3, 0.2, 0.1),
		(0.2, 0.6, 0.8, 0.6, 0.2),
		(0.3, 0.8, 1.0, 0.8, 0.3),
		(0.2, 0.6, 0.8, 0.6, 0.2),
		(0.1, 0.2, 0.3, 0.2, 0.1),
	)
	for offset_y in range(-2, 3):
		for offset_x in range(-2, 3):
			x = grid_x + offset_x
			y = grid_y + offset_y
			if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
				grid[y][x] = max(grid[y][x], brush[offset_y + 2][offset_x + 2])


def grid_from_coords(coordinates):
	grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
	for x, y in coordinates:
		apply_brush(grid, x // CELL_SIZE, y // CELL_SIZE)
	return grid


def interpolate_coords(coordinates):
	if len(coordinates) < 3 or len(coordinates) > 50:
		return coordinates
	interpolated = []
	factor = 100 // len(coordinates)
	for first, second in zip(coordinates, coordinates[1:]):
		for step in range(factor):
			ratio = step / factor
			interpolated.append((
				int(first[0] + (second[0] - first[0]) * ratio),
				int(first[1] + (second[1] - first[1]) * ratio),
			))
	interpolated.append(coordinates[-1])
	return interpolated


def scaled_coords(coordinates):
	if not coordinates:
		return []
	if len(coordinates) == 1:
		return [(WIDTH // 2, HEIGHT // 2)]
	left = min(x for x, _ in coordinates)
	right = max(x for x, _ in coordinates)
	top = min(y for _, y in coordinates)
	bottom = max(y for _, y in coordinates)
	center_x = (left + right) // 2
	center_y = (top + bottom) // 2
	scale = min(WIDTH // (right - left + 1), HEIGHT // (bottom - top + 1))
	return [
		(
			int((x - center_x) * scale * 0.8 + WIDTH // 2),
			int((y - center_y) * scale * 0.8 + HEIGHT // 2),
		)
		for x, y in coordinates
	]


def draw_screen(display, grid, inference_grid, font):
	display.fill(BLACK)
	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			value = int(255 * grid[y][x])
			pg.draw.rect(display, (value, value, value), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
	predictions = get_predictions(inference_grid)
	best = max(predictions)
	for index, probability in enumerate(predictions):
		color = GREEN if probability == best else WHITE
		label = font.render(f"{index}: {probability:.2f}", True, color)
		display.blit(label, (10, 10 + index * 20))
		pg.draw.rect(display, color, (70, 10 + index * 20, probability * 50, 13))
	message = font.render("Press c to clear the grid", True, WHITE)
	display.blit(message, (30, HEIGHT - 60))
	pg.display.flip()


async def main():
	init_layers()
	pg.init()
	display = pg.display.set_mode((WIDTH, HEIGHT))
	font = pg.font.SysFont(None, 24)
	coordinates = []
	last_inference = []
	clock = pg.time.Clock()
	running = True

	while running:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				running = False
			elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
				coordinates.append(event.pos)
			elif event.type == pg.MOUSEMOTION and event.buttons[0]:
				coordinates.append(event.pos)
			elif event.type == pg.KEYDOWN and event.key == pg.K_c:
				coordinates.clear()
				last_inference = []

		if coordinates:
			coordinates = interpolate_coords(coordinates)
			last_inference = scaled_coords(coordinates)
		draw_screen(display, grid_from_coords(coordinates), grid_from_coords(last_inference), font)
		clock.tick(60)
		await asyncio.sleep(0)

	pg.quit()


if __name__ == "__main__":
	asyncio.run(main())

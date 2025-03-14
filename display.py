import pygame as pg
import random

RES = 28
SCALEFACTOR = 28
WIDTH = RES * SCALEFACTOR
HEIGHT = RES * SCALEFACTOR
NUMTEST = 10000
OUTPUTS = 10

pg.init()

screen = pg.display.set_mode((WIDTH, HEIGHT))

class Picture:
    def __init__(self, tokens):
        self.label = tokens[0]
        self.preds = []
        self.pixels = [[0] * RES for i in range(RES)]
        for i in range(RES):
            for j in range(RES):
                self.pixels[i][j] = tokens[i * RES + j + 1]
    
    def setPreds(self, tokens):
        self.preds = tokens[0:-1]
    
def display(picture):
    correct = False
    for i in range(RES):
        for j in range(RES):
            pg.draw.rect(screen, (picture.pixels[i][j], picture.pixels[i][j], picture.pixels[i][j]), (j * SCALEFACTOR, i * SCALEFACTOR, SCALEFACTOR, SCALEFACTOR))
    font = pg.font.SysFont("arial", 30)
    for i in range(OUTPUTS):
        label = font.render(str(i) + ": " + str(picture.preds[i]), True, (0, 0, 255) if picture.preds[i] != max(picture.preds) else (255, 255, 0))
        screen.blit(label, (30, 30 * (i + 1)))
    font = pg.font.SysFont("arial", 200)
    label = font.render(str(picture.label), True, (255, 255, 0))
    screen.blit(label, (WIDTH - 150, 10))
    hi = -1
    pred = -1
    for n, v in enumerate(picture.preds):
        if v > hi:
            hi = v
            pred = n
    label = font.render(str(pred), True, (255, 255, 0))
    screen.blit(label, (30, 330))
    font = pg.font.SysFont("arial", 70)
    correct = pred == picture.label
    label = font.render("Correct" if correct else "Wrong", True, (0, 255, 0) if correct else (255, 0, 0))
    screen.blit(label, (100, HEIGHT - 150))
    pg.display.update()
    return correct

pictures = []

print("Reading data...")
f = open("mnist_test.txt", "r")
for line in f:
    tokens = line.split(',')
    for i in range(len(tokens)):
        tokens[i] = int(tokens[i])
    pictures.append(Picture(tokens))
f.close()

f2 = open("preds.txt", "r")
t = 0
for line in f2:
    tokens = line.split(' ')
    for i in range(len(tokens) - 1):
        tokens[i] = float(tokens[i])
    pictures[t].setPreds(tokens)
    t += 1
print("Finished reading data.")

display(pictures[random.randint(0, NUMTEST - 1)])
running = True
while running:
    pg.time.wait(100 if display(pictures[random.randint(0, NUMTEST - 1)]) else 1000)
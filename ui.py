import pygame
import subprocess
import sys
from tkinter import Tk, filedialog
from pathlib import Path

GRID_SIZE = 10
WINDOW_SIZE = 280
ROWS = COLS = 28
HEADER_HEIGHT = 40
RESULT_HEIGHT = 40
WINDOW_HEIGHT = HEADER_HEIGHT + WINDOW_SIZE + RESULT_HEIGHT

grid = [[(0, 0, 0) for _ in range(COLS)] for _ in range(ROWS)]

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 22)
result_text = "No result yet"
model_button = pygame.Rect(5, 5, 270, 30)
selected_model = None
running = True

def select_model():
    global selected_model

    root = Tk()
    root.withdraw()

    filename = filedialog.askopenfilename(
        title="Select model",
        initialdir=".",
        filetypes=[
            ("NumPy model weights", "*.npz"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if filename:
        selected_model = filename
        eval_result()

for y in range(ROWS):
    for x in range(COLS):
        pygame.draw.rect(
            screen,
            grid[y][x],
            (x * GRID_SIZE, HEADER_HEIGHT + y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        )

def invert_color(color):
    return tuple(255 - c for c in color)

def invert_grid():
    for y in range(ROWS):
        for x in range(COLS):
            grid[y][x] = invert_color(grid[y][x])
            pygame.draw.rect(
                screen,
                grid[y][x],
                (x * GRID_SIZE, HEADER_HEIGHT + y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            )

def eval_result():
    global result_text

    values = ",".join(
        str(sum(grid[y][x]) // 3)
        for y in range(ROWS)
        for x in range(COLS)
    )

    args = [
        sys.executable,
        "network.py",
        f"--input={values}",
        "--type=raw",
    ]

    if selected_model is not None:
        args.append(f"--model={selected_model}")

    result = subprocess.run(
        args,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        result_text = "Network error: " + result.stderr
    else:
        result_text = result.stdout.strip()

def draw_ui():
    pygame.draw.rect(
        screen,
        (25, 25, 25),
        (0, 0, WINDOW_SIZE, HEADER_HEIGHT)
    )

    pygame.draw.rect(screen, (70, 70, 70), model_button)

    model_name = (
        Path(selected_model).name
        if selected_model is not None
        else "Default model (weights.npz)"
    )

    model_label = font.render(
        f"Model: {model_name}",
        True,
        "white"
    )

    screen.blit(model_label, (10, 10))

    pygame.draw.rect(
        screen,
        "black",
        (0, HEADER_HEIGHT + WINDOW_SIZE, WINDOW_SIZE, RESULT_HEIGHT)
    )

    text_surface = font.render(result_text[:35], True, "white")
    text_rect = text_surface.get_rect(
        center=(
            WINDOW_SIZE // 2,
            HEADER_HEIGHT + WINDOW_SIZE + RESULT_HEIGHT // 2
        )
    )

    screen.blit(text_surface, text_rect)

while running:
    draw_ui()
    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONUP:
            if model_button.collidepoint(event.pos):
                select_model()
                continue

            mouse_x, mouse_y = event.pos
            grid_x = mouse_x // GRID_SIZE
            grid_y = (mouse_y - HEADER_HEIGHT) // GRID_SIZE

            if not (0 <= grid_x < COLS and 0 <= grid_y < ROWS):
                continue

            rect_x = grid_x * GRID_SIZE
            rect_y = HEADER_HEIGHT + grid_y * GRID_SIZE

            if grid[grid_y][grid_x] == (0, 0, 0):
                col = (255, 255, 255)
            else:
                col = (0, 0, 0)

            pygame.draw.rect(screen, col, (rect_x, rect_y, GRID_SIZE, GRID_SIZE))
            grid[grid_y][grid_x] = col
            eval_result()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                invert_grid()

pygame.quit()

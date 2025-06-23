# mini_arcade_hand.py – Mini‑arcade (Flappy & Dino) + controle por mão
# ---------------------------------------------------------------------------
# Requisitos:
#   pip install pygame ultralytics opencv-python-headless pyautogui numpy
# Coloque seus sprites na pasta "assets" ou ajuste ASSET_PATH abaixo.
# O modelo YOLO deve ser treinado para detecção de keypoints de mão (21 pontos) e
# salvo em "best.pt" (ou ajuste MODEL_PATH).
# ---------------------------------------------------------------------------

import os, sys, random, time
from collections import deque

import pygame
import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO

# =================== CONFIGURAÇÃO DE ASSETS ===================
ASSET_PATH  = "assets"                       # pasta com as imagens
DINO_IMG    = os.path.join(ASSET_PATH, "dino.png")
CACTO_IMG   = os.path.join(ASSET_PATH, "cacto.png")
FLAPPY_IMG  = os.path.join(ASSET_PATH, "flappy.png")
PIPE_IMG    = os.path.join(ASSET_PATH, "arvore.png")
BG_IMG      = os.path.join(ASSET_PATH, "floresta.png")
MODEL_PATH  = "best.pt"                      # modelo YOLO

# =============== INICIALIZAÇÃO PYGAME ===============
pygame.init()
WIDTH, HEIGHT = 640, 480
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mini Arcade: Flappy, Dino & Hand Control")
CLOCK = pygame.time.Clock()
FONT  = pygame.font.SysFont(None, 36)

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, BLUE  = (34, 177, 76), (0, 162, 232)

# =============== CARREGAR SPRITES ===============

def load_alpha(path: str):  
    """Carrega imagem com canal alfa. Se não existir, devolve placeholder colorido."""
    try:
        return pygame.image.load(path).convert_alpha()
    except FileNotFoundError:
        surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        surf.fill((255, 0, 255, 160))
        return surf

dino_sprite   = load_alpha(DINO_IMG)
cacto_sprite  = load_alpha(CACTO_IMG)
flappy_sprite = load_alpha(FLAPPY_IMG)
pipe_sprite   = load_alpha(PIPE_IMG)

bg_sprite = None
if os.path.exists(BG_IMG):
    bg_raw   = pygame.image.load(BG_IMG).convert()
    bg_sprite = pygame.transform.scale(bg_raw, (WIDTH, HEIGHT))

# =============== YOLO & CAMERA (HAND TRACKING) ===============
model      = YOLO(MODEL_PATH)
cap        = cv2.VideoCapture(0)
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()
smooth_queue       = deque(maxlen=5)
last_press_time    = 0
press_interval     = 0.5  # segundos
frame_count        = 0

# ===== Funções de mão =====

def is_finger_extended(tip, pip, mcp):
    tip = np.array(tip[:2]); pip = np.array(pip[:2]); mcp = np.array(mcp[:2])
    return np.linalg.norm(tip - mcp) > np.linalg.norm(pip - mcp)

def is_hand_open(keypoints):
    fingers = {
        'indicador': (8, 6, 5),
        'medio':     (12,10,9),
        'anelar':    (16,14,13),
        'mindinho':  (20,18,17)
    }
    extended = sum(is_finger_extended(keypoints[tip], keypoints[pip], keypoints[mcp])
                   for tip, pip, mcp in fingers.values())
    return extended >= 4  # mão aberta se >=4 dedos estendidos

# =============== VARIÁVEIS DE ESTADO DO JOGO ===============
state = 'menu'  # 'menu' | 'flappy' | 'dino'

# ---------- MENU ----------
flappy_btn = pygame.Rect(WIDTH//2 - 120, HEIGHT//2 - 60, 240, 50)
dino_btn   = pygame.Rect(WIDTH//2 - 120, HEIGHT//2 + 20, 240, 50)

# ---------- FLAPPY VAR ----------
GRAV_F, JUMP_F = 0.45, -8
BIRD_X, PIPE_GAP = 80, 120
PIPE_SPEED = 3
PIPE_W     = pipe_sprite.get_width()

def reset_flappy():
    global bird_y, bird_vel, pipes, score_f, pipe_timer
    bird_y, bird_vel = HEIGHT//2, 0
    pipes, score_f, pipe_timer = [], 0, 0

reset_flappy()

# ---------- DINO VAR ----------
BASELINE = HEIGHT - 30  # chão visível
JUMP_D, GRAV_D = -12, 0.6
DINO_X = 60

def reset_dino():
    global dino_bottom, dino_vel, obstacles, score_d, obs_timer
    dino_bottom, dino_vel = BASELINE, 0
    obstacles, score_d, obs_timer = [], 0, 0

reset_dino()

# ---------- UTIL ----------
render_text = lambda txt, pos, col=BLACK: SCREEN.blit(FONT.render(txt, True, col), pos)

# =============== LOOP PRINCIPAL ===============
running = True
while running:
    CLOCK.tick(60)
    frame_count += 1

    # ======== YOLO HAND DETECTION (a cada 2 frames) ========
    hand_open = None; avg_x = avg_y = None
    ret, frame = cap.read()
    if ret and frame_count % 2 == 0:
        h, w, _ = frame.shape
        results = model.predict(source=frame, show=False, save=False, stream=True)
        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                keypoints = r.keypoints.data[0].cpu().numpy().tolist()
                if len(keypoints) >= 21:
                    hand_open = is_hand_open(keypoints)

                    # posição dedo indicador (ponto 8)
                    x, y = keypoints[8][:2]
                    smooth_queue.append((x, y))
                    avg_x = sum(p[0] for p in smooth_queue) / len(smooth_queue)
                    avg_y = sum(p[1] for p in smooth_queue) / len(smooth_queue)

                    # MENU -> controla mouse
                    if state == 'menu' and avg_x is not None:
                        screen_x = int(((w - avg_x) / w) * screen_w)  # espelhado
                        screen_y = int((avg_y / h) * screen_h)
                        pyautogui.moveTo(screen_x, screen_y, duration=0.05)
                        if not hand_open and time.time() - last_press_time > press_interval:
                            pyautogui.click()
                            last_press_time = time.time()
                    # JOGO -> pula ao fechar a mão (sem mexer mouse)
                    elif state in ('flappy', 'dino'):
                        if hand_open is not None and not hand_open and time.time()-last_press_time>press_interval:
                            if state == 'flappy':
                                bird_vel = JUMP_F
                            elif state == 'dino':
                                if dino_bottom >= BASELINE - 1:
                                    dino_vel = JUMP_D
                            last_press_time = time.time()
                    break  # só usa a primeira mão detectada
        # opcional: mostrar janela debug
        cv2.imshow('Hand Pose', frame)
        if cv2.waitKey(1) == 27:  # ESC fecha app inteiro
            running = False

    # ======== EVENTOS PYGAME ========
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if state == 'menu' and e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if flappy_btn.collidepoint(e.pos):
                reset_flappy(); state = 'flappy'
            elif dino_btn.collidepoint(e.pos):
                reset_dino();  state = 'dino'
        if state == 'flappy':
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                bird_vel = JUMP_F
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                state = 'menu'
        if state == 'dino':
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_UP):
                if dino_bottom >= BASELINE - 1:
                    dino_vel = JUMP_D
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                state = 'menu'

    # ======== RENDERIZAÇÃO ========
    if state == 'menu':
        SCREEN.fill(WHITE)
        pygame.draw.rect(SCREEN, GREEN, flappy_btn)
        pygame.draw.rect(SCREEN, BLUE,  dino_btn)
        render_text("Jogar Flappy Bird", (flappy_btn.x + 20, flappy_btn.y + 10), WHITE)
        render_text("Jogar Dino Game",  (dino_btn.x + 25, dino_btn.y + 10), WHITE)
        render_text("Controle por mão: aponte & feche pra clicar", (10, HEIGHT - 60))
        render_text("ESC fecha o programa", (10, HEIGHT - 30))

    # ---------- FLAPPY ----------
    elif state == 'flappy':
        SCREEN.blit(bg_sprite, (0, 0)) if bg_sprite else SCREEN.fill((135, 206, 235))
        bird_vel += GRAV_F; bird_y += bird_vel
        # gerar canos
        pipe_timer += 1
        if pipe_timer > 90:
            pipe_timer = 0
            gap_y = random.randint(100, HEIGHT - 100)
            pipes.append(pygame.Rect(WIDTH, 0, PIPE_W, gap_y - PIPE_GAP // 2))
            pipes.append(pygame.Rect(WIDTH, gap_y + PIPE_GAP // 2, PIPE_W, HEIGHT - gap_y))
        for p in pipes[:]:
            p.x -= PIPE_SPEED
            pipe_img = pygame.transform.scale(pipe_sprite, (p.width, p.height))
            if p.y == 0:
                pipe_img = pygame.transform.flip(pipe_img, False, True)
            SCREEN.blit(pipe_img, p)
            if p.right < 0:
                pipes.remove(p)
                if p.y == 0:
                    score_f += 1
        bird_rect = flappy_sprite.get_rect(center=(BIRD_X, bird_y))
        SCREEN.blit(flappy_sprite, bird_rect)
        if bird_rect.top < 0 or bird_rect.bottom > HEIGHT or any(bird_rect.colliderect(p) for p in pipes):
            reset_flappy(); state = 'menu'
        render_text(f"Pontos: {score_f}", (10, 10))

    # ---------- DINO ----------
    elif state == 'dino':
        SCREEN.blit(bg_sprite, (0, 0)) if bg_sprite else SCREEN.fill(WHITE)
        pygame.draw.line(SCREEN, BLACK, (0, BASELINE), (WIDTH, BASELINE), 2)
        dino_vel += GRAV_D; dino_bottom += dino_vel
        if dino_bottom > BASELINE: dino_bottom, dino_vel = BASELINE, 0
        dino_rect = dino_sprite.get_rect(midbottom=(DINO_X, dino_bottom))
        SCREEN.blit(dino_sprite, dino_rect)
        obs_timer += 1
        if obs_timer > 80:
            obs_timer = 0
            ob_rect = cacto_sprite.get_rect(bottomleft=(WIDTH, BASELINE))
            obstacles.append(ob_rect)
        for ob in obstacles[:]:
            ob.x -= 6
            SCREEN.blit(cacto_sprite, ob)
            if ob.right < 0:
                obstacles.remove(ob); score_d += 1
        if any(dino_rect.colliderect(ob) for ob in obstacles):
            reset_dino(); state = 'menu'
        render_text(f"Pontos: {score_d}", (10, 10))

    pygame.display.flip()

# ---------- FINALIZA ----------
cap.release(); cv2.destroyAllWindows(); pygame.quit()

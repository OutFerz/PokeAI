import os
import glob
import time
import cv2
import numpy as np
from sb3_contrib import RecurrentPPO
from src.environment.pokemon_env import PokemonYellowEnv

# --- CONFIGURACIÓN ---
MODEL_DIR = "experiments/poke_lstm_v1/models"
ROM_PATH = "roms/PokemonYellow.gb"
SCALE = 3
FPS = 60  # Velocidad objetivo real
FRAMES_PER_ACTION = 24 # 🔥 DEBE COINCIDIR con el .tick() de tu pokemon_env.py

# --- ESTÉTICA GAMEBOY ---
GB_CASE = (180, 180, 180)    
GB_SCREEN_BORDER = (100, 100, 100)
GB_DPAD_DARK = (40, 40, 40)  
GB_DPAD_LIGHT = (60, 60, 60) 
GB_BTN_PURPLE = (100, 50, 100) 
GB_BTN_PURPLE_L = (130, 70, 130) 
COLOR_TEXT = (50, 50, 50)    
COLOR_ON_NEON = (0, 255, 255) 

PANEL_W = 300
DPAD_CENTER = (100, 150)
DPAD_SIZE = 35 
BTN_A_CENTER = (240, 120)
BTN_B_CENTER = (190, 150)
BTN_RADIUS = 25

def get_latest_model():
    list_of_files = glob.glob(f'{MODEL_DIR}/*.zip')
    if not list_of_files: return None
    return max(list_of_files, key=os.path.getctime)

# --- FUNCIONES DE DIBUJO ---
def draw_gb_button_circle(panel, center, radius, base_color, light_color, text, is_pressed):
    cv2.circle(panel, center, radius, base_color, -1)
    inner_color = COLOR_ON_NEON if is_pressed else light_color
    cv2.circle(panel, center, radius - 5, inner_color, -1)
    text_color = (0,0,0) if is_pressed else (200,200,200)
    cv2.putText(panel, text, (center[0]-10, center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

def draw_gb_dpad(panel, center, size, pressed_idx):
    cx, cy = center
    s = size
    cv2.rectangle(panel, (cx-s, cy-s//3), (cx+s, cy+s//3), GB_DPAD_DARK, -1)
    cv2.rectangle(panel, (cx-s//3, cy-s), (cx+s//3, cy+s), GB_DPAD_DARK, -1)
    c_up = COLOR_ON_NEON if pressed_idx == 3 else GB_DPAD_LIGHT
    c_down = COLOR_ON_NEON if pressed_idx == 0 else GB_DPAD_LIGHT
    c_left = COLOR_ON_NEON if pressed_idx == 1 else GB_DPAD_LIGHT
    c_right = COLOR_ON_NEON if pressed_idx == 2 else GB_DPAD_LIGHT
    pad = 5
    cv2.rectangle(panel, (cx-s+pad, cy-s//3+pad), (cx-s//3, cy+s//3-pad), c_left, -1)
    cv2.rectangle(panel, (cx+s//3, cy-s//3+pad), (cx+s-pad, cy+s//3-pad), c_right, -1)
    cv2.rectangle(panel, (cx-s//3+pad, cy-s+pad), (cx+s//3-pad, cy-s//3), c_up, -1)
    cv2.rectangle(panel, (cx-s//3+pad, cy-s//3+pad), (cx+s//3-pad, cy+s-pad), c_down, -1)
    cv2.rectangle(panel, (cx-s//3+pad, cy-s//3+pad), (cx+s//3-pad, cy+s//3-pad), GB_DPAD_LIGHT, -1)

def draw_gamepad_panel(pressed_btn_idx, height, is_lstm=False):
    panel = np.full((height, PANEL_W, 3), GB_CASE, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (20, height), GB_SCREEN_BORDER, -1)
    title = "LSTM CORTEX" if is_lstm else "NEURAL INPUT"
    cv2.putText(panel, title, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    draw_gb_dpad(panel, DPAD_CENTER, DPAD_SIZE, pressed_btn_idx)
    draw_gb_button_circle(panel, BTN_A_CENTER, BTN_RADIUS, GB_BTN_PURPLE, GB_BTN_PURPLE_L, "A", pressed_btn_idx == 4)
    draw_gb_button_circle(panel, BTN_B_CENTER, BTN_RADIUS, GB_BTN_PURPLE, GB_BTN_PURPLE_L, "B", pressed_btn_idx == 5)
    return panel

def main():
    print("--- STREAM GAME BOY VISUALIZER (SMOOTH CINEMA MODE) ---")
    
    # Render_mode='rgb_array' para que PyBoy no abra su ventana, solo nosotros
    env = PokemonYellowEnv(ROM_PATH, render_mode="rgb_array") 
    
    current_model_path = None
    model = None
    lstm_states = None 
    episode_starts = np.ones((1,), dtype=bool)
    
    # Obs inicial
    obs, _ = env.reset()

    try:
        while True:
            # 1. BUSCAR CEREBRO NUEVO
            latest_model_path = get_latest_model()
            if not latest_model_path:
                print("Esperando primer modelo...", end="\r")
                time.sleep(2)
                continue
                
            if latest_model_path != current_model_path:
                print(f"[UPDATE] Cargando: {os.path.basename(latest_model_path)}")
                # Reiniciamos el entorno visual al cargar modelo nuevo para ver desde el inicio
                obs, _ = env.reset()
                model = RecurrentPPO.load(latest_model_path, env=env)
                current_model_path = latest_model_path
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)
            
            # 2. LA IA PIENSA (1 vez cada 24 frames)
            if model:
                action_idx, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts,
                    deterministic=False
                )
            else:
                action_idx = env.action_space.sample()

            # 3. EJECUCIÓN SUAVE (Desenrollamos el loop temporal)
            # En lugar de env.step() que salta 24 frames, lo hacemos a mano paso a paso
            
            # Convertimos índice a nombre de botón (ej: 4 -> 'a')
            action_name = env.valid_actions[action_idx]
            frames_to_hold = 12            
            # 🔥 LOOP DE RENDERIZADO SUAVE (rellenamos los huecos)
            for i in range(FRAMES_PER_ACTION):
                frame_start = time.time()
                
                if i < frames_to_hold:
                    env.pyboy.button(action_name)
                
                # Avanzamos SOLO 1 frame
                env.pyboy.tick(1) 
                
                # --- DIBUJAR ---
                game_pixels = env.pyboy.screen.ndarray # PyBoy 2.0
                if game_pixels.shape[2] == 4: game_pixels = game_pixels[:, :, :3]
                game_bgr = cv2.cvtColor(game_pixels, cv2.COLOR_RGB2BGR)
                h, w, _ = game_bgr.shape
                game_view = cv2.resize(game_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
                
                # Dibujamos panel (mantenemos el botón presionado visualmente)
                gamepad_view = draw_gamepad_panel(action_idx, height=game_view.shape[0], is_lstm=True)
                final_visual = np.hstack((game_view, gamepad_view))
                cv2.imshow("indigoRL | Smooth View", final_visual)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

                # --- CONTROL DE FPS ---
                frame_time = time.time() - frame_start
                delay = (1.0 / FPS) - frame_time
                if delay > 0:
                    time.sleep(delay)
            
            # 4. ACTUALIZAR OBSERVACIÓN REAL
            # Una vez pasados los 24 frames, tomamos la foto para la siguiente decisión de la IA
            # Usamos la función interna _get_obs() porque evitamos llamar a step()
            obs = env._get_obs()
            
            # Reset de flag de episodio
            episode_starts = np.zeros((1,), dtype=bool)

    except KeyboardInterrupt: pass
    except Exception as e: print(f"Error: {e}")
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
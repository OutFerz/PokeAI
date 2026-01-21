from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from src.environment.pokemon_env import PokemonYellowEnv
import os

# --- CONFIGURACIÓN ---
ROM_PATH = "roms/PokemonYellow.gb"
SESSION_NAME = "poke_lstm_v1"
CHECKPOINT_DIR = f"experiments/{SESSION_NAME}/models"
LOG_DIR = f"experiments/{SESSION_NAME}/logs"
TOTAL_TIMESTEPS = 5000000 
NUM_CPU = 6 

# Frecuencia de guardado: Cada 20 updates (aprox cada 5-10 minutos reales)
# 12288 pasos por update * 20 = 245760 pasos
SAVE_FREQ = 245760 // NUM_CPU 

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("--- INICIANDO ENTRENAMIENTO PRO CON MEMORIA (LSTM) ---")

if __name__ == "__main__":
    # 1. Crear entorno Vectorizado (6 CPUs)
    env = make_vec_env(
        lambda: PokemonYellowEnv(ROM_PATH, render_mode='rgb_array'),
        n_envs=NUM_CPU,
        vec_env_cls=SubprocVecEnv
    )

    # 2. Configurar el Guardado Automático (Para poder ver el Watch)
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="lstm_model"
    )

    # 3. Crear modelo Recurrente (LSTM) + MultiInput
    model = RecurrentPPO(
        "MultiInputLstmPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        n_steps=2048, 
        batch_size=128,
        n_epochs=10,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            enable_critic_lstm=False, 
            lstm_hidden_size=256,
        )
    )

    # 4. Entrenar con seguridad (Ctrl+C guarda el modelo)
    print(f"Entrenando en {NUM_CPU} entornos paralelos a ~800 FPS...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            tb_log_name="LSTM_Run",
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n!!! Interrupción detectada. Guardando modelo de emergencia... !!!")
    finally:
        model.save(f"{CHECKPOINT_DIR}/final_model_lstm")
        env.close()
        print("¡Entrenamiento finalizado y guardado!")
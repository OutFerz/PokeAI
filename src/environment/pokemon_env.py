import io
import os
import random
from gymnasium import Env, spaces
import numpy as np
from pyboy import PyBoy
from skimage.transform import resize

class PokemonYellowEnv(Env):
    def __init__(self, rom_path, render_mode='rgb_array', observation_type='multi'):
        super().__init__()
        self.rom_path = rom_path
        self.render_mode = render_mode
        self.observation_type = observation_type

        # --- DIRECCIONES DE MEMORIA (Pokémon Yellow - English/International) ---
        # wEventFlags: 320 bytes de historia (bits)
        self.MEM_EVENT_FLAGS_START = 0xD747
        self.MEM_EVENT_FLAGS_END = 0xD747 + 320 
        self.MEM_MAP_ID = 0xD35E
        self.MEM_IS_IN_BATTLE = 0xD057
        self.MEM_ENEMY_HP_HIGH = 0xCFE6
        self.MEM_ENEMY_HP_LOW = 0xCFE7
        self.MEM_MY_HP_HIGH = 0xD16C
        self.MEM_MY_HP_LOW = 0xD16D
        self.MEM_PARTY_LEVELS = 0xD18C
        
        # Configuración PyBoy 2.0
        window_type = "null" if render_mode == 'rgb_array' else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window_type)
        if render_mode == 'rgb_array':
            self.pyboy.set_emulation_speed(0) 

        self.screen_width = 160
        self.screen_height = 144
        self.render_callback = None 

        self.valid_actions = ['down', 'left', 'right', 'up', 'a', 'b', 'start']
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # Observación
        self.output_shape = (3, self.screen_height, self.screen_width)
        screen_space = spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)
        
        # RAM: [X, Y, MapID, MyHP, EnemyHP, Levels, InBattle]
        ram_space = spaces.Box(low=0, high=255, shape=(7,), dtype=np.uint8)

        self.observation_space = spaces.Dict({
            'screen': screen_space,
            'ram': ram_space
        })

        # Variables de estado
        self.visited_maps = set()
        self.visited_coords = set()
        self.last_event_count = 0
        self.last_hp = 20
        self.last_party_levels = 5
        self.last_enemy_hp = 0
        self.step_count = 0
        
        # Timeout más agresivo para forzar eficiencia
        self.max_steps = 2048 * 6 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if hasattr(self, 'pyboy'): self.pyboy.stop()

        window_type = "null" if self.render_mode == 'rgb_array' else "SDL2"
        self.pyboy = PyBoy(self.rom_path, window=window_type)
        if self.render_mode == 'rgb_array': self.pyboy.set_emulation_speed(0)

        # --- CARGA DE ESTADO (Saltar Intro) ---
        state_path = "states/start.state"
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                self.pyboy.load_state(f)
        else:
            print("⚠️ AVISO: No se encontró 'states/start.state'. La IA jugará la intro.")

        # Reiniciar variables
        self.visited_maps = set()
        self.visited_coords = set()
        self.step_count = 0
        
        # Lecturas iniciales
        self.last_hp = self._read_hp()
        self.last_party_levels = self._read_party_levels()
        self.last_event_count = self._read_event_count()
        self.last_enemy_hp = self._read_enemy_hp()
        
        # Añadir mapa actual a visitados para evitar premio gratis al reinicio
        self.visited_maps.add(self.pyboy.memory[self.MEM_MAP_ID])

        return self._get_obs(), {}

    def step(self, action_idx):
        self.step_count += 1
        
        # Acción
        action = self.valid_actions[action_idx]
        self.pyboy.button(action)
        self.pyboy.tick(24) 

        if self.render_callback: self.render_callback(action_idx)

        obs = self._get_obs()
        reward = self._compute_reward()

        terminated = False
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Pantalla
        screen = self.pyboy.screen.ndarray 
        screen = resize(screen, (self.screen_height, self.screen_width), anti_aliasing=False, preserve_range=True)
        screen = screen.astype(np.uint8)
        if screen.shape[2] == 4: screen = screen[:, :, :3]
        screen = np.moveaxis(screen, 2, 0)

        # RAM
        x_coord = self.pyboy.memory[0xD362]
        y_coord = self.pyboy.memory[0xD361]
        map_id = self.pyboy.memory[self.MEM_MAP_ID]
        my_hp = self._read_hp()
        enemy_hp = self._read_enemy_hp()
        levels = self._read_party_levels()
        in_battle = self.pyboy.memory[self.MEM_IS_IN_BATTLE]

        ram_data = np.array([x_coord, y_coord, map_id, my_hp, enemy_hp, levels, in_battle], dtype=np.uint8)

        return {'screen': screen, 'ram': ram_data}

    def _compute_reward(self):
        reward = 0
        
        # 1. HISTORIA (Event Flags - El objetivo principal)
        current_event_count = self._read_event_count()
        if current_event_count > self.last_event_count:
            diff = current_event_count - self.last_event_count
            reward += (diff * 20.0) # ¡Premio masivo!
            self.last_event_count = current_event_count
            print(f"🌟 EVENTO DESBLOQUEADO ({current_event_count})")

        # 2. NUEVOS MAPAS
        map_id = self.pyboy.memory[self.MEM_MAP_ID]
        if map_id not in self.visited_maps:
            self.visited_maps.add(map_id)
            reward += 5.0
            print(f"🗺️ NUEVO MAPA: {map_id}")

        # 3. BATALLAS (Dañar al enemigo)
        is_in_battle = self.pyboy.memory[self.MEM_IS_IN_BATTLE]
        current_enemy_hp = self._read_enemy_hp()
        
        if is_in_battle:
            if self.last_enemy_hp > 0:
                damage = self.last_enemy_hp - current_enemy_hp
                if damage > 0:
                    reward += damage * 0.2 # Premia la agresividad
            self.last_enemy_hp = current_enemy_hp
        else:
            self.last_enemy_hp = 0

        # 4. NIVELES
        current_levels = self._read_party_levels()
        if current_levels > self.last_party_levels:
            reward += 5.0
            self.last_party_levels = current_levels

        # 5. EXPLORACIÓN LOCAL (Pequeña ayuda para moverse)
        x = self.pyboy.memory[0xD362]
        y = self.pyboy.memory[0xD361]
        coord = (x, y, map_id)
        if coord not in self.visited_coords:
            self.visited_coords.add(coord)
            reward += 0.05
        
        return reward

    # --- LECTORES DE MEMORIA ---
    def _read_hp(self):
        return (self.pyboy.memory[self.MEM_MY_HP_HIGH] << 8) + self.pyboy.memory[self.MEM_MY_HP_LOW]

    def _read_enemy_hp(self):
        return (self.pyboy.memory[self.MEM_ENEMY_HP_HIGH] << 8) + self.pyboy.memory[self.MEM_ENEMY_HP_LOW]
    
    def _read_party_levels(self):
        return self.pyboy.memory[self.MEM_PARTY_LEVELS] 

    def _read_event_count(self):
        # Lee el bloque exacto de flags de historia
        event_bytes = self.pyboy.memory[self.MEM_EVENT_FLAGS_START : self.MEM_EVENT_FLAGS_END]
        return sum(bin(byte).count('1') for byte in event_bytes)

    def render(self):
        return self.pyboy.screen.ndarray

    def close(self):
        if hasattr(self, 'pyboy') and self.pyboy:
            self.pyboy.stop()
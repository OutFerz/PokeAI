from pyboy import PyBoy
import os

# Ensure the folder exists
if not os.path.exists("states"):
    os.makedirs("states")

print("--- GRABADORA DE ESTADOS (PyBoy 2.0) ---")
print("1. Se abrirá el juego. Juega la intro (Usa Flechas, A=Z, B=X, Start=Enter).")
print("2. Cuando aparezcas en la habitación de Ash y tengas el control: CIERRA LA VENTANA.")
print("3. El estado se guardará automáticamente en 'states/start.state'.")

# Initialize PyBoy (starts automatically in v2.0)
pyboy = PyBoy("roms/PokemonYellow.gb", window="SDL2")

try:
    # Infinite loop that keeps the game running until you close the window
    while pyboy.tick():
        pass
except KeyboardInterrupt:
    pass

# On loop exit (window close), we save
print("\nGuardando estado...")
with open("states/start.state", "wb") as f:
    pyboy.save_state(f)

pyboy.stop()
print("✅ ¡Estado guardado exitosamente en 'states/start.state'!")
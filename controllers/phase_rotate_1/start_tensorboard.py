import subprocess
import webbrowser
import time
import sys
import os

# You need to do `pip install tensorboard`

# 1. Descobre a pasta exata onde este script 'start_tensorboard.py' está salvo
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define o caminho para a pasta onde os logs realmente estão (ajuste o nome da pasta se necessário)
# Aqui ele vai procurar a pasta 'tensorboard' dentro de 'phase_rotate_1'
log_dir_absoluto = os.path.join(script_dir, "tensorboard")

print(f"Buscando logs em: {log_dir_absoluto}")

# 3. Monta o comando usando a API interna
comando = [
    sys.executable,
    "-c",
    "from tensorboard import main; main.run_main()"
]

# Passa o caminho absoluto gerado dinamicamente
argumentos = ["--logdir", log_dir_absoluto]

process = subprocess.Popen(comando + argumentos)
print("TensorBoard iniciado com sucesso via API interna!")

# Wait a little for startup
time.sleep(10)

# Open browser automatically
webbrowser.open("http://localhost:6006")

# Keep process alive
process.wait()
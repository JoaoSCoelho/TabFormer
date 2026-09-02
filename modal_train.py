import modal
import subprocess
import os

# 1. O Computador: Limpo e moderno! 
# O Modal vai usar o Python padrão atual (3.11+) e as versões mais recentes das bibliotecas.
# Sem limites de versão, sem hacks de infraestrutura.
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", 
        "transformers", 
        "pandas", 
        "numpy", 
        "scikit-learn", 
        "tqdm"
    )
    .add_local_dir(local_path=".", remote_path="/root/tabformer")
)

app = modal.App("tabformer-gpt2-train", image=image)

# 2. Conectamos o nosso "pen-drive" virtual (onde os dados de cartão já estão salvos)
data_volume = modal.Volume.from_name("tabformer-data")

# 3. Configuramos a GPU e disparamos a função nativamente
@app.function(
    gpu="A100-40GB", 
    timeout=86400, # Permite rodar por até 24 horas
    volumes={"/root/data": data_volume} 
)
def train_model():
    # Entramos na pasta do projeto
    os.chdir("/root/tabformer")
    
    print("🚀 Iniciando treinamento acelerado na GPU com ambiente moderno...")
    
    # O seu comando de execução intacto
    subprocess.run([
        "python", "main.py", 
        "--do_train", 
        "--lm_type", "gpt2", 
        "--field_ce", 
        "--data_type", "card",
        "--data_root", "/root/data/credit_card/",
        "--output_dir", "/root/data/output-gpt",
        "--save_steps", "2000"
    ], check=True)
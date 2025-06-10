import wandb
import copy
import os 
import yaml
from train import TrainPipeline
with open("config.yaml", "r") as file:
    base_config = yaml.safe_load(file)

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
sweep_config = {
    "method": "grid",
    "metric": {"name": "eval_f1_score", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [3e-5, 5e-5, 7e-5]},
        "batch_size": {"values": [16, 32]},
        "epochs": {"values": [2, 3, 4]}
    }
}

PROJECT = base_config["wandb"]["project_name"]

sweep_id = wandb.sweep(sweep_config, project=PROJECT)
print(f"✅ Sweep created with ID: {sweep_id}")

def train_sweep():
    wandb.init()
    sweep_params = wandb.config

    updated_config = copy.deepcopy(base_config)

    model_key = updated_config['train']['model']  
    updated_config[model_key]['learning_rate'] = float(sweep_params.learning_rate)
    updated_config[model_key]['batch_size'] = sweep_params.batch_size
    updated_config[model_key]['epochs'] = sweep_params.epochs

    TrainPipeline(config=updated_config).run()

wandb.agent(sweep_id, train_sweep, count=18)

import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_RNA  
from dataset_rna2 import get_new_rna_dataloader  
from utils import train, evaluate

# Argument parser
parser = argparse.ArgumentParser(description="CSDI for RNA Sequencing")
parser.add_argument("--config", type=str, default="base.yaml", help="Path to config file")
parser.add_argument("--device", default="cuda:0", help="Device for training (e.g., cpu or cuda:0)")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--testmissingratio", type=float, default=0.1, help="Missing data ratio for testing")
parser.add_argument("--modelfolder", type=str, default="", help="Folder to load pre-trained model from")
parser.add_argument("--nsample", type=int, default=100, help="Number of samples for evaluation")

args = parser.parse_args()

# Load config file
config_path = os.path.join("config", args.config)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Update test missing ratio in config
config["model"]["test_missing_ratio"] = args.testmissingratio

# Create an output folder for results
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/rna_{current_time}/"
os.makedirs(foldername, exist_ok=True)

# Save the configuration for reference
config_save_path = os.path.join(foldername, "config.json")
with open(config_save_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Config saved to: {config_save_path}")

# Create DataLoaders
train_loader, valid_loader, test_loader = get_new_rna_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

# Initialize model
model = CSDI_RNA(config, args.device).to(args.device)

# Train or Load Model
if args.modelfolder == "":
    print("Training new model...")
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model_path = os.path.join("./save", args.modelfolder, "model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f"Loaded pre-trained model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
print("Evaluation complete. Results saved in:", foldername)
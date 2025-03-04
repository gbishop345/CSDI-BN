import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_RNA  
from dataset_rna import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI for RNA Sequencing")
parser.add_argument("--config", type=str, default="base.yaml", help="Path to config file")
parser.add_argument("--device", default="cuda:0", help="Device for training")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--testmissingratio", type=float, default=0.1, help="Missing data ratio for testing")
parser.add_argument("--modelfolder", type=str, default="", help="Folder to load pre-trained model from")
parser.add_argument("--nsample", type=int, default=100, help="Number of samples for evaluation")

args = parser.parse_args()

# Load config
config_path = "config/" + args.config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["test_missing_ratio"] = args.testmissingratio

# Create output folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/rna_{current_time}/"
os.makedirs(foldername, exist_ok=True)

with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# Create data loaders
train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

# Init model
model = CSDI_RNA(config, args.device).to(args.device)

# Train or Load
if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load(f"./save/{args.modelfolder}/model.pth"))

# Evaluate
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
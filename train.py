import argparse
from datetime import datetime
import tqdm
import yaml
import numpy as np
import torch

from utils.metrics import Logger
from utils.misc import existsfolder, get_config
from utils.data import HarborfrontDataset, get_transforms
from torch.utils.data import DataLoader
from model import get_model

if __name__ == "__main__":
    #CLI
    parser = argparse.ArgumentParser("Train XAI-MU model")
    #Positionals
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    #Optional
    parser.add_argument("--device", default="cuda:0", help="Which device to prioritize")
    parser.add_argument("--output", default="./assets/", help="Where to save the model weights")
    parser.add_argument("--verbose", default=False, action='store_true', help="Enable verbose status printing")
    args = parser.parse_args()        

    print("\n########## TRACE AND ERASE ##########")
    #Load configs
    cfg = get_config(args.config)

    print("\n########## PREPARING DATA ##########")
    #Setup preprocessing steps
    label_transforms = get_transforms("label")
    train_transforms = get_transforms("train", cfg["augmentation"])
    valid_transforms = get_transforms("valid", cfg["augmentation"])

    print("\n### CREATING TRAINING DATASET")
    train_dataset = HarborfrontDataset(
        data_split=cfg["data"]["train"], 
        root=cfg["data"]["root"], 
        transform=train_transforms, 
        target_transform=label_transforms, 
        classes=cfg["data"]["classes"], 
        binary_labels=cfg["data"]["binary_cls"],
        classwise=cfg["data"]["classwise"], 
        device="cpu", 
        verbose=True)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
    )

    print("\n### CREATING VALIDATION DATASET")
    valid_dataset = HarborfrontDataset(
        data_split=cfg["data"]["valid"], 
        root=cfg["data"]["root"], 
        transform=train_transforms, 
        target_transform=label_transforms, 
        classes=cfg["data"]["classes"], 
        binary_labels=cfg["data"]["binary_cls"],
        classwise=cfg["data"]["classwise"], 
        device="cpu", 
        verbose=True)
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["training"]["batch_size"],
    )

    print("\n### DATALOADER SANITY CHECK")
    dummy_sample = next(iter(train_dataloader))
    print(f"Train: Input Tensor = {dummy_sample[0].shape}")
    print(f"Train: Label Tensor = {dummy_sample[1].shape}")
    dummy_sample = next(iter(valid_dataloader))
    print(f"Valid: Input Tensor = {dummy_sample[0].shape}")
    print(f"Valid: Label Tensor = {dummy_sample[1].shape}")

    print("\n########## BUILDING MODEL ##########")
    print(f"MODEL ARCH: {cfg['model']['arch']}")
    model = get_model(cfg, args.device)

    #Define optimizer
    print(f"OPTIMIZER: torch.optim.SGD")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=5e-4
        )

    #Define loss
    print(f"LOSS: {cfg['training']['loss']}")
    if cfg["training"]["loss"] == "huber":
        loss_fn = torch.nn.HuberLoss(delta=2.0)
    elif cfg["training"]["loss"] == "l1":
        loss_fn = torch.nn.L1Loss()
    elif cfg["training"]["loss"] == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise Exception(f"UNKNOWN LOSS: '{cfg['training']['loss']}' must be one of the following: 'l1', 'mse', 'huber' ")


    print("\n########## LOCAL OUTPUTS ##########")
    #Create output folder
    out_folder = f'{args.output}/{datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}/'
    print(f"Saving weights and logs at '{out_folder}'")
    existsfolder(out_folder)
    existsfolder(out_folder+"/weights")

    #Save copy of config
    cfg["folder_name"] = out_folder
    with open(out_folder + "/config.yaml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    #Logging
    if cfg["evaluation"]["classwise_metrics"]:
        train_logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"], classwise_metrics=train_dataset.classes)
        valid_logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"], classwise_metrics=train_dataset.classes)
    else:
        train_logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"])
        valid_logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"])

    #Plotting for Validation
    if cfg["wandb"]["plotting"]:
        extra_plots = {}
        from utils.wandb_plots import conf_matrix_plot
        from functools import partial
        extra_plots[f"conf_plot"] = conf_matrix_plot
        if cfg["evaluation"]["classwise_metrics"]:
            for i,c in enumerate(cfg["data"]["classes"]):
                extra_plots[f"conf_plot_{c}"] = partial(conf_matrix_plot, idx=i)

    print("\n########## TRAINING MODEL ##########")
    print(f"Logging progress every: {cfg['training']['log_freq']} batches ({cfg['training']['log_freq']*cfg['training']['batch_size']} samples)")
    print(f"Running evaluation every: {cfg['evaluation']['eval_freq']} batches ({cfg['evaluation']['eval_freq']*cfg['training']['batch_size']} samples)")
    best_model=10000 #We want the loss to be lower than this before we save anything
    itteration = 0
    for epoch in tqdm.tqdm(range(cfg["training"]["epochs"]), unit="Epoch", desc="Epochs"):
        #Train
        model.train()
        running_loss = 0
        for i, batch in tqdm.tqdm(enumerate(train_dataloader), unit="Batch", desc="Training", leave=False, total=len(train_dataloader)):
            #Count batches for eval checkpoints
            itteration += 1
            
            #Reset gradients (redundant but sanity check)
            optimizer.zero_grad()
            
            #Seperate batch
            inputs, labels = batch
            
            #Forward
            outputs = model(inputs)
            
            #Calculate loss
            loss = loss_fn(outputs, labels)
            loss.backward()
            running_loss += loss.item() / batch.shape[0]

            #Propogate error
            optimizer.step()

            #Store prediction
            train_logger.add_prediction(outputs.detach().to("cpu").numpy(), labels.detach().to("cpu").numpy())

            #Log training stats
            if i % cfg["training"]["log_freq"] == 0:
                logs = train_logger.log(
                    clear_buffer=True,
                    prepend='train',
                    xargs={
                        "loss": running_loss/cfg["training"]["log_freq"]
                    },
                )
                running_loss = 0

            #Log validation stats
            if i % cfg["evaluation"]["eval_freq"] == 0 or i >= len(train_dataloader)-1:
                #Proceed to Validation
                model.eval()
                with torch.no_grad():
                    valid_loss = 0
                    valid_logger.clear_buffer()
                    for j, val_batch in tqdm.tqdm(enumerate(valid_dataloader), unit="Batch", desc="Validating", leave=False, total=len(valid_dataloader)):
                        #Seperate batch
                        val_inputs, val_labels = val_batch
                        
                        #Forward
                        outputs = model(val_inputs)

                        #Calculate loss
                        val_loss += loss_fn(outputs, val_labels).item() / val_batch.shape[0]

                        #Store prediction
                        valid_logger.add_prediction(outputs.detach().to("cpu").numpy(), val_labels.detach().to("cpu").numpy())

                    #Log validation metrics
                    validation_loss = val_loss / len(valid_dataloader)
                    val_logs = valid_logger.log(
                        clear_buffer=True,
                        prepend='valid',
                        extras=extra_plots,
                        xargs={
                            "loss": validation_loss,
                            "itteration": itteration,
                        },
                    )
                    val_loss = 0
                    print(f"Validation after {itteration}")
                    print(val_logs)
                    
                    #Save best model
                    if validation_loss < best_model:
                        torch.save(model.state_dict(), out_folder + "/weights/" + f'checkpoint-itt-{itteration}-loss{validation_loss}.pt')

                model.train()
        #Save Model
        torch.save(model.state_dict(), out_folder + "/weights/" + f'checkpoint-epoch-{epoch}.pt')

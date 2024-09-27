import argparse
from datetime import datetime
import tqdm
import yaml
import numpy as np
import torch

from utils.metrics import Logger
from utils.misc import existsfolder, get_config
from utils.data import get_dataset, get_dataloader, get_transforms
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

    print("\n########## COUNT-EXPLAIN-REMOVE ##########")
    #Load configs
    cfg = get_config(args.config)

    print("\n########## PREPARING DATA ##########")
    #Setup preprocessing steps
    label_transforms = get_transforms("label")
    train_transforms = get_transforms("train", cfg["augmentation"])
    valid_transforms = get_transforms("valid", cfg["augmentation"])

    print("\n### CREATING TRAINING DATASET")

    # TODO make dataloader
    train_dataloader = get_dataloader(cfg, get_dataset(cfg, "train", train_transforms, label_transforms))
    valid_dataloader = get_dataloader(cfg, get_dataset(cfg, "valid", valid_transforms, label_transforms))

    #print example batch (sanity check)
    dummy_sample = next(iter(train_dataloader))
    print(f"Input Tensor = {dummy_sample[0].shape}")
    print(f"Label Tensor = {dummy_sample[1].shape}")

    print("\n########## BUILDING MODEL ##########")
    # TODO instansiate model
    model = get_model(cfg, args.device)

    #Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=0.9,
        weight_decay=5e-4
        )

    #Define loss
    if cfg["training"]["loss"] == "huber":
        loss_fn = torch.nn.HuberLoss(delta=2.0)
    elif cfg["training"]["loss"] == "l1":
        loss_fn = torch.nn.L1Loss()
    elif cfg["training"]["loss"] == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise Exception(f"UNKNOWN LOSS: '{cfg['training']['loss']}' must be one of the following: 'l1', 'mse', 'huber' ")

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
        train_logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"], classwise_metrics=cfg["data"]["classes"])
        valid_logger = Logger(cfg, out_folder=out_folder, metrics=cfg["evaluation"]["metrics"], classwise_metrics=cfg["data"]["classes"])
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
    for epoch in tqdm.tqdm(range(cfg["training"]["epochs"]), unit="Epoch", desc="Epochs"):
        #Train
        model.train()
        running_loss = 0
        for i, batch in tqdm.tqdm(enumerate(train_dataloader), unit="Batch", desc="Training", leave=False, total=len(train_dataloader)):

            #Reset gradients (redundant but sanity check)
            optimizer.zero_grad()
            
            #Seperate batch
            inputs, labels = batch
            
            #Forward
            outputs = model(inputs)
            
            #Calculate loss
            loss = loss_fn(outputs, labels)
            loss.backward()
            running_loss += loss.item()

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
                        "loss": running_loss/len(train_dataloader)
                    },
                )
                running_loss = 0

            #Log validation stats
            if i % cfg["evaluation"]["eval_freq"] == 0 or i >= len(train_dataloader)-1:
                #Proceed to Validation
                with torch.no_grad():
                    valid_loss = 0
                    valid_logger.clear_buffer()
                    for j, val_batch in tqdm.tqdm(enumerate(valid_dataloader), unit="Batch", desc="Validating", leave=False, total=len(valid_dataloader)):
                        #Seperate batch
                        val_inputs, val_labels = val_batch
                        
                        #Forward
                        outputs = model(val_inputs)

                        #Calculate loss
                        val_loss += loss_fn(outputs, val_labels).item()

                        #Store prediction
                        valid_logger.add_prediction(outputs.detach().to("cpu").numpy(), val_labels.detach().to("cpu").numpy())

                    #Log validation metrics
                    val_logs = valid_logger.log(
                        clear_buffer=True,
                        prepend='valid',
                        extras=extra_plots,
                        xargs={
                            "loss": val_loss / len(valid_dataloader)
                        },
                    )
                    val_loss = 0
                    print(val_logs)
        
        #Save Model
        torch.save(model.state_dict(), out_folder + "/weights/" + f'{cfg["model"]["arch"]}-{cfg["model"]["task"]}-f{cfg["model"]["arch"]}-E{epoch}.pt')

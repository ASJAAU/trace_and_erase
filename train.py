import argparse
from datetime import datetime
import tqdm
import yaml
import math
import torch

from utils.metrics import Logger
from utils.misc import existsfolder, get_config
from utils.data import HarborfrontDataset, get_transforms
from torch.utils.data import DataLoader
from utils.model import get_model

# Because we do evaluation based on several conditions evaluation has been made into a function
def validate(model, valid_dataloader, valid_logger, extra_plots={}, xargs={}, prepend='valid'):
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
            valid_loss += loss_fn(outputs, val_labels).item()

            #Store prediction
            valid_logger.add_prediction(outputs.detach().to("cpu").numpy(), val_labels.detach().to("cpu").numpy())

        #Log validation metrics
        validation_loss = valid_loss / len(valid_dataloader)

        #Parse additional external info
        extra_info = {
            f"{prepend}_loss": validation_loss,
        }
        for key,value in xargs.items():
            extra_info[key] = value

        #Log 
        val_logs = valid_logger.log(
            clear_buffer=True,
            prepend=prepend,
            extras=extra_plots,
            xargs=extra_info,
        )
        return val_logs, validation_loss

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
        in_channels=cfg["model"]["in_channels"], 
        classes=cfg["data"]["classes"], 
        binary_labels=cfg["data"]["binary_cls"],
        classwise=cfg["evaluation"]["classwise_metrics"], 
        device=args.device, 
        verbose=True)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    if args.verbose:
        print("Train data sanity check")
        dummy_sample = next(iter(train_dataloader))
        print(f"Train: Input Tensor = {dummy_sample[0].shape}")
        print(f"Train: Label Tensor = {dummy_sample[1].shape}\n values: \n{dummy_sample[1]}")

    print("\n### CREATING VALIDATION DATASET")
    valid_dataset = HarborfrontDataset(
        data_split=cfg["data"]["valid"], 
        root=cfg["data"]["root"], 
        transform=train_transforms, 
        target_transform=label_transforms,
        in_channels=cfg["model"]["in_channels"], 
        classes=cfg["data"]["classes"], 
        binary_labels=cfg["data"]["binary_cls"],
        classwise=cfg["evaluation"]["classwise_metrics"], 
        device=args.device, 
        verbose=True)
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["training"]["batch_size"],
    )

    if args.verbose:
        print("Validation data sanity check")
        dummy_sample = next(iter(valid_dataloader))
        print(f"Valid: Input Tensor = {dummy_sample[0].shape}")
        print(f"Valid: Label Tensor = {dummy_sample[1].shape}\n values: \n{dummy_sample[1]}")

    print("\n########## BUILDING MODEL ##########")
    print(f"MODEL ARCH: {cfg['model']['arch']}")
    model = get_model(cfg, args.device)


    #Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=5e-4
        )
    
    #Define learning-rate schedule
    if cfg["training"]["lr_decay_step"] == "epoch":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=len(train_dataloader), 
            gamma=cfg["training"]["lr_decay"],
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfg["training"]["lr_decay_step"], 
            gamma=cfg["training"]["lr_decay"],
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
    extra_plots = {}
    if cfg["wandb"]["enabled"] and cfg["wandb"]["plotting"]: 
        from utils.wandb_plots import conf_matrix_plot
        from functools import partial
        extra_plots[f"conf_plot"] = conf_matrix_plot
        if cfg["evaluation"]["classwise_metrics"]:
            for i,c in enumerate(cfg["data"]["classes"]):
                extra_plots[f"conf_plot_{c}"] = partial(conf_matrix_plot, idx=i)

    print("\n########## TRAINING MODEL ##########")
    print(f"Logging progress every: {cfg['training']['log_freq']} batches ({cfg['training']['log_freq']*cfg['training']['batch_size']} samples)")
    print(f"Progress will be logged {len(train_dataloader) / cfg['training']['log_freq']} times per epoch")
    print(f"Running evaluation every: {cfg['evaluation']['eval_freq']} batches ({cfg['evaluation']['eval_freq']*cfg['training']['batch_size']} samples)")
    print(f"In addition to end of epoch evaluation, evaluation will be performed {len(train_dataloader) / cfg['evaluation']['eval_freq']} times per epoch")
    
    best_model=10000 #We want the loss to be lower than this before we save anything
    itteration = 0
    
    #Train
    for epoch in tqdm.tqdm(range(cfg["training"]["epochs"]), unit="Epoch", desc="Epochs"):
        model.train()
        running_loss = 0
        for it, batch in tqdm.tqdm(enumerate(train_dataloader), unit="Batch", desc="Training", leave=False, total=len(train_dataloader)):
            #Count batches for eval checkpoints
            itteration += 1
            
            #Reset gradients (redundant but sanity check)
            optimizer.zero_grad()
            
            #Seperate batch
            inputs, labels = batch
            
            #Forward
            outputs = model(inputs.to(args.device))

            #Ignore class in loss function

            #XAI
            #Not yet implemented

            #Calculate loss
            loss = loss_fn(outputs, labels)
            loss.backward()
            running_loss += loss.item()

            #Propogate error
            optimizer.step()

            #Store prediction
            train_logger.add_prediction(outputs.detach().to("cpu").numpy(), labels.detach().to("cpu").numpy())

            #Log training stats
            if itteration % cfg["training"]["log_freq"] == 0 and it != 0:
                logs = train_logger.log(
                    clear_buffer=True,
                    prepend='train',
                    xargs={
                        "loss": running_loss/cfg["training"]["log_freq"],
                        "running_loss": running_loss,
                        "lr": lr_scheduler.get_last_lr(),
                        "itteration": itteration,
                    },
                )
                running_loss = 0

            #Run and log validation stats
            if it % cfg["evaluation"]["eval_freq"] == 0 and it != 0:
                extra_logging = {
                    "itteration": itteration,
                }
                log, validation_loss = validate(model, valid_dataloader, valid_logger, extra_plots=extra_plots, xargs=extra_logging)
                if validation_loss < best_model:
                    best_model = validation_loss
                    torch.save(model.state_dict(), out_folder + "/weights/" + f'checkpoint-itt-{itteration}-loss{validation_loss:.2f}.pt')
                model.train()
            elif it+1 >= len(train_dataloader):
                extra_logging = {
                    "itteration": itteration,
                }
                log, validation_loss = validate(model, valid_dataloader, valid_logger, extra_plots=extra_plots, xargs=extra_logging)
                if validation_loss < best_model:
                    best_model = validation_loss
                    torch.save(model.state_dict(), out_folder + "/weights/" + f'checkpoint-epoch-{epoch}-loss{validation_loss:.2f}.pt')
                model.train()
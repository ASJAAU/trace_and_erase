import timm

def get_model(cfg, device):
    model = timm.create_model(
            cfg["model"]["arch"], 
            pretrained=False, 
            in_chans=3, 
            num_classes = len(cfg["data"]["classes"]),).to(device)
    
    #Load optional weights
    if cfg["model"]["weights"] is not None:
        try:
            model.load_state_dict(torch.load(cfg["model"]["weights"]))
            print(f"Loaded weights from: '{cfg["model"]["weights"]}'")
        except:
            raise Exception(f"Failed to load weights from '{cfg["model"]["weights"]}'")
    return model
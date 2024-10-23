import timm
import torch


def get_model(cfg, device):
    model = timm.create_model(
            cfg["model"]["arch"], 
            pretrained=cfg["model"]["weights"], 
            in_chans=cfg["model"]["in_channels"],
            num_classes = len(cfg["data"]["classes"]),
            ).to(device)
    
    #Load optional weights
    if cfg["model"]["weights"] is not None:
        try:
            model.load_state_dict(torch.load(cfg["model"]["weights"]))
            print(f"Loaded weights from: '{cfg['model']['weights']}'")
        except:
            raise Exception(f"Failed to load weights from '{cfg['model']['weights']}'")
    
    if cfg["model"]["intermediate_outputs"] is not None:
        model.intermediate_outputs={}  
        print("Exposing intermediate representations")
        for layer_name in cfg['model']['intermediate_outputs']:
            layer = eval(layer_name)
            def hook(module,input,output):
                model.intermediate_outputs[layer] = output.detach()
            layer.register_forward_hook(hook)
            print(f"Hook for '{layer}', registered")
    return model

if __name__ == '__main__':
    from utils.misc import get_config
    cfg = get_config("configs/base.yaml")
    model = get_model(cfg, 'cpu')

    #Test inference
    dummy_batch_size=4
    dummy_data = torch.rand(dummy_batch_size,cfg["model"]["in_channels"],224,224)

    print(f"Input: {dummy_data.shape}")
    outputs = model(dummy_data)
    print(f"Output:\n", outputs.detach().shape)
    if cfg["model"]["intermediate_outputs"] is not None:
        print(f"Intermediates:\n")
        for intermediate in model.intermediate_outputs:
            print(model.intermediate_outputs[intermediate].shape)
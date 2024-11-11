# Based on :
# Pytorch SIDU: https://github.com/MarcoParola/pytorch_sidu/sidu.py
# and
# Original SIDU: https://github.com/satyamahesh84/SIDU_XAI_CODE/blob/main/SIDU_XAI.py

import torch
import numpy as np

#testing
import time

def kernel(tensor: torch.Tensor, kernel_width: float = 0.25) -> torch.Tensor:
    """
    Kernel function for computing the weights of the differences.

    Args:
        vector (torch.Tensor): 
            The difference tensor.
        kernel_width (float, optional): 
            The kernel width. Defaults to 0.1.

    Returns:
        torch.Tensor: 
            The weights.
    """
    return torch.sqrt(torch.exp(-(tensor ** 2) / kernel_width ** 2))

def uniqness_measure(masked_predictions: torch.Tensor) -> torch.Tensor:
    r""" Compute the uniqueness measure

    Args:
        masked_predictions: torch.Tensor
            The predicitons from masked featuremaps

    Returns:
        uniqueness: torch.Tensor
            The uniqueness measure
    """

    # Compute pairwise distances between each prediction vector
    distances = torch.cdist(masked_predictions, masked_predictions)
    # Compute sum along the last two dimensions to get uniqueness measure for each mask
    uniqueness = torch.sum(distances, dim=-1)
    #Normalize uniqueness
    return torch.nn.functional.normalize(uniqueness, dim=0)

def similarity_differences(orig_predictions: torch.Tensor, masked_predictions: torch.Tensor):
    r""" Compute the similarity differences

    Args:
        orig_predictions: torch.Tensor
            The original predictions
        masked_predictions: torch.Tensor
            The masked predictions

    Returns:
         : torch.Tensor
            The weights
        diff: torch.Tensor
            The differences
    """
    diff = torch.abs(masked_predictions - orig_predictions).mean(axis=1)
    weights = kernel(diff)
    return weights, diff

def generate_masks(img_size: tuple, feature_map: torch.Tensor, device="cpu") -> torch.Tensor:
    r""" Generate masks from the feature map

    Args:
        img_size: tuple
            The size of the input image [H,W]
        feature_map: torch.Tensor
            The feature map from the model [C,H,W]

    Returns:
        masks: torch.Tensor
            The generated masks
    """

    N = feature_map.shape[0]

    #Masks placeholder of N masks
    masks = torch.zeros(size=(N,*img_size), device=feature_map.device)

    #Threshold to binary values
    test = torch.nn.functional.threshold(feature_map, 0.15, 1.0, 0.0)

    #Upsample
    test = torch.nn.functional.interpolate(test.unsqueeze(0), size=(img_size[0], img_size[1]), mode='bilinear', align_corners=True)

    return masks#, feature_map.detach().clone() #Grid

def sidu(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    r""" SIDU SImilarity Difference and Uniqueness method
    Note: The original implementation processes B,H,W,C as per TF standard, where Torch uses BCHW
    
    Args:
        model: torch.nn.Module
            The model to be explained 
        image: torch.Tensor
            The input image to be explained 
        device: torch.device, optional
            The device to use. Defaults to torch.device("cpu")
    Returns:
        saliency_maps: torch.Tensor
            The saliency maps

    """
    
    # Storage for return values
    maps = []
    predictions = []
    predictions.extend(model(inputs).detach())

    #Disable AutoGrad
    with torch.no_grad():
        #Forward pass to extract base predictions and intermediary featuremaps
        print(f"Sidu input: {inputs.shape}")
        for i, input in enumerate(inputs):
            bigtstep = time.time()
            #Take ith sample from batch
            input = inputs[i].unsqueeze(0) #Keep batch dimension
            #Run Sidu on all available intermediary layers
            for inter in model.intermediate_outputs:
                orig_feature_map = model.intermediate_outputs[inter][i].detach()

                #Generate masks
                masks= generate_masks((input.shape[-2],input.shape[-1]), orig_feature_map, device=device)
                N = masks.shape[0] #N = Channels at intermediary position
            
                #Predictions (explain_SIDU in original TF)
                masked_predictions = []

                #Masked batches
                batch_size = 50

                #apply masks to all channels
                masked = masks.unsqueeze(1) * input

                #Process masked predictions
                for j in range(0,N,batch_size):
                    masked_predictions.append(model(masked[j:min(j+batch_size,N)].to(device)).detach())

                #align predictions
                masked_predictions = torch.cat(masked_predictions, dim=0)

                #Compute weights and differences
                weights, diff = similarity_differences(predictions[i], masked_predictions)
                
                #Compute uniqueness to infer sample uniqueness
                uniqueness = uniqness_measure(masked_predictions)
                
                #Apply weight to uniqueness
                weighted_uniqueness = uniqueness * weights

                #Generated weighted saliency map
                saliency_map = masks * weighted_uniqueness.unsqueeze(dim=-1). unsqueeze(dim=-1)
            
                # reduce the saliency maps to a single map by summing over the masks dimension
                saliency_map = saliency_map.sum(dim=0)
                saliency_map /= N

                #Add to list of maps
                maps.append(saliency_map.cpu().squeeze(0).numpy())
            print(f"Total Sidu: {time.time()-bigtstep}\n")
            

        #Return entire batch
        return predictions, maps
  
    
if __name__ == '__main__':
    import torch
    import timm
    from torchvision.transforms import v2 as torch_transforms
    from torchvision.io import read_image
    from visualize import visualize_prediction
    from data import get_transforms
    from model import get_model
    from misc import get_config

    device="cuda:0"

    # Get training image transforms
    transforms = get_transforms("train")
    img_paths = ["assets/test_image.jpg"]*8

    #Load image / images    
    input = [read_image(img_path) for img_path in img_paths]

    #Preprocess images
    input = [transforms(img) for img in input]
    input = torch.stack(input, dim=0)

    #Load Pretrained model
    config = get_config("configs/base.yaml")
    model = get_model(config, device)

    # Generate SIDU Heatmaps
    predictions, heatmaps = sidu(model, input.to(device), device=device)

    #Show prediction
    for i in range(input.shape[0]):
        visualize_prediction(input[i].permute(1, 2, 0).numpy(), None, None, [heatmaps[i]], classes=["All"], blocking=True)

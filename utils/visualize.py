import matplotlib
from matplotlib.pyplot import subplots, show, figure
from matplotlib import patches, colors, rc
import numpy as np

def visualize_heatmap(image, heatmap, cmap='jet'):
    fig, ax = subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('tight')
    ax.axis('off')

    #Original image
    ax.imshow(heatmap, cmap=cmap, alpha=1.0)
    ax.axis('off')
    return fig

def visualize_prediction(image, predictions=None, groundtruth=None, heatmaps=None, centers=None, bbox=None, classes=None, cmap='jet'):
    #Get color map (for annotations)
    colormap = [
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
    ]
    
    #Heatmaps?
    len_heatmaps = len(heatmaps) if heatmaps is not None else 0
    
    #Make figure
    fig, axs = subplots(1+len_heatmaps, 1, figsize=(8,6 * (len_heatmaps+1)))

    #Format initial image
    if len_heatmaps<1:
        in_im_fig = axs
    else:
        in_im_fig = axs[0]


    #Remove image ticks
    in_im_fig.axis('off')

    #Original image
    in_im_fig.imshow(image, cmap='gray', vmin=0, vmax=255)
    #in_im_fig.set_title("Input Image")

    #Check if there is predictions
    if predictions is not None:
        #Check if class names exist
        if classes is None:
            classes = [f'{i}' for i in range(len(predictions))]
        else:
            assert len(predictions) <= len(classes), "The length of 'classes' argument needs to be equal or larger than length of predictions"

        #Write prediction - cmultilabel classification/regression
        text = [f"{classes[i]}: {f'{float(predictions[i])}'} {f'({int(groundtruth[i])})' if groundtruth is not None else ''}" for i in range(len(predictions))]
        
        #Put the text in a box
        in_im_fig.text(0.01, 0.99, "\n".join(text), transform=in_im_fig.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left', 
                bbox={
                    'boxstyle':'round', 
                    'facecolor':'wheat', 
                    'alpha':0.8
                    }
                )

    #Check for groundtruth bounding boxes
    if bbox is not None:
        for i in range(len(bbox)):
            c, x1, y1, x2,y2 = bbox[i]
            w = x2 - x1
            h = y2 - y1
            in_im_fig.add_patch(patches.Rectangle((x1,y1),w,h,linewidth=1, edgecolor=colormap[c], facecolor=colormap[c][:3]+(0.15,)))#'none'))
    
    #Check for groundtruth center points
    if centers is not None:
        for i in range(len(centers)):
            c, x1, y1 = centers[i]
            in_im_fig.add_patch(patches.Circle((x1,y1) ,radius=2, linewidth=1, facecolor=colormap[c]))

    #Draw heatmaps
    if heatmaps is not None:
        for i in range(len(heatmaps)):
            #Remove graph ticks
            axs[i+1].axis('off')
            #Set image
            axs[i+1].set_title(f"Heatmap: {classes[i]}")
            axs[i+1].imshow(image, aspect='equal', cmap='gray')
            #Set heatmap
            hmap = axs[i+1].imshow(heatmaps[i], cmap=cmap, alpha=0.5)
            #Plot Colorbar / colormap
            cax = axs[i+1].inset_axes([0.2, -0.04, 0.6, 0.04], transform=axs[i+1].transAxes)
            cax.axis('off')
            fig.colorbar(hmap, cax=cax, orientation='horizontal')
            

    #Compress padding
    fig.tight_layout()
    
    #Return figure for possible saving
    return fig


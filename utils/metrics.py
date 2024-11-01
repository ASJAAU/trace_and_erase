import numpy as np
from functools import partial

### LOGGING
class Logger:
    def __init__(self, cfg, out_folder, metrics=[], classwise_metrics=[]) -> None:
        #Save local copies of relevant variables
        self.metrics = {}
        #Retrieve metrics
        for metric in metrics:
            self.metrics.update(get_metric(metric, classwise_metrics))
        #Establish output files and classes
        self.output_path = out_folder
        self.classes = cfg["data"]["classes"]

        #Init wandb
        if cfg["wandb"]["enabled"]:
            self.wandb = wandb_logger(
                cfg,
                output_path=out_folder,
                )
        else:
            self.wandb = None

        #Buffer for predictions
        self.preds = []
        self.labels= []

    def clear_buffer(self):
        self.preds = []
        self.labels= []

    def log(self, xargs={}, clear_buffer=True, prepend='', extras=None):
        preds = np.concatenate(self.preds, axis=0)
        labels= np.concatenate(self.labels, axis=0)

        # Create step log
        to_log = {}

        #Add xargs to log
        for k,v in xargs.items():
            to_log[k]=v

        # Apply metrics
        for name, fn in self.metrics.items():
            to_log[f'{prepend}_{name}'] = fn(preds, labels)

        #Optional Extra metrics functions (For plots etc.)
        if extras is not None:
            for name, fn in extras.items():
                to_log[f'{prepend}_{name}'] = fn(preds, labels)

        #upload to wandb
        if self.wandb is not None:
            self.wandb.log(to_log)

        #Clear buffer post logging
        if clear_buffer:
            self.clear_buffer()

        #return results
        return to_log
    
    def add_prediction(self, prediction, label):
        self.preds.append(prediction)
        self.labels.append(label)

def wandb_logger(cfg, output_path="./wandb"):
    import wandb
    if cfg["wandb"]["resume"] is not None:
        wandb_logger = wandb.init(
            project=cfg["wandb"]["project_name"],
            config=cfg,
            tags=cfg["wandb"]["tags"],
            resume="must",
            id=cfg["wandb"]["resume"],
            entity=cfg["wandb"]["entity"],
            notes=cfg["experiment"]["notes"],
            dir=output_path,
            force=True,
            
        )
    else:
        wandb_logger = wandb.init(
            project=cfg["wandb"]["project_name"],
            config=cfg,
            tags=cfg["wandb"]["tags"],
            dir=output_path,
            entity=cfg["wandb"]["entity"],
        )
    return wandb_logger
    
### HELPER FUNCTIONS FOR LOGGING
def get_metric(metric, classwise=[]):
    metrics = {}
    #MAE
    if metric == "mae":
        metrics["MAE"] = mae
        for i,c in enumerate(classwise):
            metrics[f"MAE_{c}"] = partial(mae, idx=i)
    #MSE
    elif metric == "mse":
        metrics["MSE"] = mse
        for i,c in enumerate(classwise):
            metrics[f"MSE_{c}"] = partial(mse, idx=i)

    #RMSE
    elif metric == "rmse":
        metrics["RMSE"] = rmse
        for i,c in enumerate(classwise):
            metrics[f"RMSE_{c}"] = partial(rmse, idx=i)
    #MAPE
    elif metric == "mape":
        metrics["MAPE"] = mape
        for i,c in enumerate(classwise):
            metrics[f"MAPE_{c}"] = partial(mape, idx=i)
    #MASE    
    elif metric == "mase":
        metrics["MASE"] = mase
        for i,c in enumerate(classwise):
            metrics[f"MASE_{c}"] = partial(mase, idx=i)
    else:
        print(f"UNRECOGNIZED METRIC: {metric}, IGNORED")
    return metrics
    
def get_wandb_plots(names):
    import utils.wandb_plots as wandplots
    plots = {}
    for name in names:
        plots[name] = getattr(wandplots)
    return plots

#Metrics functions
# ALL metrics functions expect the same two inputs
# preds = 2D numpy array of Batch * predictions
# labels = 2D numpy array of Batch * Labels
# idx = an index for index specific metric calculation

def mae(preds, labels, idx=None):
    if idx is None: #Calculating total MAE
        return np.mean(abs(preds-labels))
    else: #Calculating class specific MAE
        return np.mean(abs(preds[:,idx]-labels[:,idx]))
    
def mape(preds, labels, idx=None):
    if idx is None: #Calculating total MAPE
        return np.mean((abs(preds-labels)/labels)*100)
    else: #Calculating class specific MAPE
        return np.mean((abs(preds[:,idx]-labels[:,idx])/labels)*100)
    
def mase(preds, labels, idx=None):
    if idx is None: #Calculating total MASE
        return np.mean(abs(preds-labels)) / np.mean(abs(labels-np.tile(np.mean(labels, axis=0),(labels.shape[0],1))))
    else: #Calculating class specific MASE
        return np.mean(abs(preds[:,idx]-labels[:,idx])) / np.mean(abs(labels[:,idx]-np.tile(np.mean(labels[:,idx], axis=0),(labels.shape[0],1))))

def rmse(preds, labels, idx=None):
    if idx is None: #Calculating total RMSE
        return np.sqrt(np.mean(abs(preds-labels)**2))
    else: #Calculating class specific RMSE
        return np.sqrt(np.mean(abs(preds[:,idx]-labels[:,idx])**2))

def mse(preds, labels, idx=None):
    if idx is None: #Calculating total RMSE
        return np.mean(abs(preds-labels**2))
    else: #Calculating class specific RMSE
        return np.mean(abs(preds[:,idx]-labels[:,idx]**2))

def r2(preds, labels, idx=None):
    return None

def HmCvr(heatmap, mask):
    return np.sum(np.multiply(heatmap,mask)) / np.sum(heatmap)

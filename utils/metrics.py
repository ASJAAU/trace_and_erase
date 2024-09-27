import numpy as np

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
    raise NotImplemented
    return metrics
    
def get_wandb_plots(names):
    import utils.wandb_plots as wandplots
    plots = {}
    for name in names:
        plots[name] = getattr(wandplots)
    return plots

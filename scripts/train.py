import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import wandb

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    
    wandb.init(project="neuro_lab", config=dict(cfg))
    
    model = instantiate(cfg.model)
    
    datamodule = instantiate(cfg.data)
    
    print(f"Modèle instancié : {type(model)}")
    print(f"Architecture : \n{model}")
    

if __name__ == "__main__":
    main()

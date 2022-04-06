import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=".", config_name="config")
def my_app(cfg : DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    print(cfg.env)

if __name__ == "__main__":
    my_app()
from jaad.jaad import parse_sgnet_args
from dataloaders.data_utils import build_data_loader


if __name__ == "__main__":
    jaad_args = parse_sgnet_args()
    jaad_dataloader = build_data_loader(jaad_args)

    print(next(iter(jaad_dataloader))['target_y'])
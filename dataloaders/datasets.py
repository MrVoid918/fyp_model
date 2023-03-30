from .jaad_data_layer import JAADDataLayer
from .pie_data_layer import PIEDataLayer

def build_dataset(args, phase):
    print(args.dataset)
    if args.dataset in ['JAAD']:
        data_layer = JAADDataLayer
    elif args.dataset in ['PIE']:
        data_layer = PIEDataLayer
    return data_layer(args, phase)
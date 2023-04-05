from base_configs import parse_base_args

__all__ = ['parse_sgnet_args']

def parse_sgnet_args():
    parser = parse_base_args()
    parser.add_argument('--dataset', default='JAAD', type=str)
    parser.add_argument('--data_root', default='../JAAD/data/JAAD-JAAD_2.0', type=str)
    parser.add_argument('--lr', default=5e-04, type=float)
    parser.add_argument('--model', default='SGNet_CVAE', type=str)
    parser.add_argument('--bbox_type', default='cxcywh', type=str)
    parser.add_argument('--normalize', default='none', type=str)
    # parser.add_argument('--normalize', default='none', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--enc-steps', default=15, type=int)
    parser.add_argument('--dec-steps', default=45, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nu', default=0.0, type=float)
    parser.add_argument('--sigma', default=1.5, type=float)
    parser.add_argument('--FPS', default=30, type=int)
    parser.add_argument('--min_bbox', default=[0,0,0,0], type=list)
    parser.add_argument('--max_bbox', default=[1920, 1080, 1920, 1080], type=list)
    parser.add_argument('--K', default=20, type=int)
    parser.add_argument('--DEC_WITH_Z', default=True, type=bool)
    parser.add_argument('--LATENT_DIM', default=32, type=int)
    parser.add_argument('--pred_dim', default=4, type=int)
    parser.add_argument('--input-dim', default=8, type=int)
    parser.add_argument('--velocity', action='store_true', help='include velocity as input')
    
    parser.add_argument('--save-weights', type=str, default='weights/pred_weights_random.tar')
    parser.add_argument('--show-result', action='store_true', )
    parser.add_argument('--show-interval', default=5, type=int, help='show result every n epochs')


    return parser.parse_args()

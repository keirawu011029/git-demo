import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str, default='MI_RAP_every_momentum')
parser.add_argument("--model_dir", type=str, default='./models')
parser.add_argument("--result_dir", type=str, default='./result_cifar10/find_best_vit_cifar10')
parser.add_argument("--result_file_name", type=str, default=None)
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--targerted_attack', action='store_true')
parser.add_argument('--model_num', type=int, default=40)
parser.add_argument('--total_step', type=int, default=1)
parser.add_argument('--step_size', type=float, default=2/255)
parser.add_argument('--reverse_step_size', type=float, default=0.1/255)
parser.add_argument('--reverse_step', type=int, default=5)
parser.add_argument('--late_start', type=int, default=15)
parser.add_argument('--inner_step_size', type=int, default=250)
parser.add_argument("--model_archs", nargs='+',default='resnetat xcitat')

parser.add_argument('--vit_model_dir', type=str, default='')
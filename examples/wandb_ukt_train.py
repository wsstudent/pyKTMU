import argparse
from wandb_train import main, unlearning_arg_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[unlearning_arg_parser])
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--model_name", type=str, default="ukt")
    parser.add_argument("--emb_type", type=str, default="stoc_qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--final_fc_dim", type=int, default=256)
    parser.add_argument("--final_fc_dim2", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--loss1", type=float, default=0.5)
    parser.add_argument("--loss2", type=float, default=0.5)
    parser.add_argument("--loss3", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=50)


    parser.add_argument("--use_CL", type= int, default=0)
    parser.add_argument("--cl_weight", type= float, default=0.02)
    parser.add_argument("--use_uncertainty_aug", type= int, default=1)

    parser.add_argument("--atten_type", type= str, default="w2")
    parser.add_argument("--use_wandb", type= int, default=1)
    parser.add_argument("--add_uuid", type= int, default=1)

    args = parser.parse_args()


    params = vars(args)
    main(params)
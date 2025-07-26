import argparse
from wandb_retrain import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Standard model and dataset parameters
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="dkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # Hyperparameters
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # Wandb and logging
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    # === New parameters for retraining on a specific dataset ===
    # Switch to enable retraining
    parser.add_argument(
        "--retrain_from_scratch",
        action="store_true",
        help="Enable retraining from scratch on the retain set.",
    )
    # Strategy used to generate the retain set
    parser.add_argument(
        "--unlearn_strategy",
        type=str,
        default="random",
        help="The unlearning strategy for which to use the retain set.",
    )
    # Forget ratio used during preprocessing
    parser.add_argument(
        "--forget_ratio",
        type=float,
        default=0.2,
        help="The forget ratio used during data preprocessing.",
    )
    
    args = parser.parse_args()

    params = vars(args)
    main(params)

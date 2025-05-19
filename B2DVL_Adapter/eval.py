import argparse
from eval_configure import EvalConfig
from eval_dataset import B2DVLEvalDataset
from eval_workers import EvalTaskDistributor
from stats import save_stats

def main(args):

    print("======== Configuration Starts ========")
    config = EvalConfig(args.config_dir)
    config.display_config()
    print("========= Configuration Ends =========")

    dataset = B2DVLEvalDataset(image_dir=config.CONFIGS['B2D_DIR'], 
                               vqa_dir=config.CONFIGS['INFERENCE_RESULT_DIR'],
                               original_vqa_dir=config.CONFIGS['ORIGINAL_VQA_DIR'])

    # Create the task distributor
    distributor = EvalTaskDistributor(
        dataset,
        num_workers=args.num_workers,
        outdir=args.out_dir,
        configs=config,
        num_sample=args.num_sample
    )
    
    distributor.distribute_tasks()
    save_stats(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval code for Bench2DriveLM dataset")
    parser.add_argument(
        "-c", "--config_dir", type=str, required=True,
        help="Path to the config file."
    )

    parser.add_argument(
        "-w", "--num_workers", type=int, default=4,
        help="Number of workers for parallel processing, defaul = 4"
    )

    parser.add_argument(
        "-o", "--out_dir", type=str, required=True,
        help="Path to the output of this code"
    )

    parser.add_argument(
        "-n", "--num_sample", type=int, required=False, default=-1,
        help="Number of frames evaluated each scenario, if not given, evaluate all."
    )

    args = parser.parse_args()
    
    main(args)
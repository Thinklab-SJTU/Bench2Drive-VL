import argparse
from b2dvl_dataset import B2DVLDataset
from inference_workers import InferTaskDistributor
from infer_configure import InferConfig
from io_utils import *

def main(args):
    transform = None
    model = args.model
    model_path = args.model_path
    clean_cache()

    print("======== Configuration Starts ========")
    if args.wp_code == None:
        # print_error("waypoint decode method not set, set to raw (no special tokenize).")
        args.wp_code = "raw"
    config = InferConfig(args.config_dir)
    config.display_config()
    print("========= Configuration Ends =========")

    dataset = B2DVLDataset(args.image_dir, args.vqa_dir, transform=transform)

    # Create the task distributor
    distributor = InferTaskDistributor(
        dataset, 
        transform=transform, 
        model=model, 
        model_path=model_path,
        num_workers=args.num_workers,
        outdir=args.out_dir,
        wp_code=args.wp_code,
        configs=config
    )
    
    distributor.distribute_tasks()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference code for Bench2Drive-VL dataset")
    parser.add_argument("--model", type=str, required=True, help="VLM name used.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model you use.")
    parser.add_argument("--config_dir", type=str, required=False, help="Path to the config file, if empty, we use default config.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory (original B2D dataset)")
    parser.add_argument("--vqa_dir", type=str, required=True, help="Path to the VQA JSON directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel processing")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output of this code")
    parser.add_argument("--wp_code", type=str, required=False, help="Decode method of waypoint (if used)")

    args = parser.parse_args()
    
    main(args)

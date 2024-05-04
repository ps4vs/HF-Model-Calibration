import os
import wandb
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='google/vit-base-patch16-224', help='huggingface hub ckpt location')
parser.add_argument('--dataset', type=str, default="imagenet-1k", help='huggingface hub dataset location')
parser.add_argument('--split', type=str, default="validation", help="which dataset split to load")
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--num_images', type=int, default=2000, help='number of images to be used for calibration')
args = parser.parse_args()
    
if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = "0aa429d933365fa8e91048dd017a5410ef2a8c51"
    os.environ['HF_TOKEN'] = "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"

    image_processor, model = utils.prepare_model(args.model)
    dataset = utils.prepare_dataset(args.dataset, args.split, args.num_images)

    save_dir = f'./{args.model}_{args.dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    logits_np, labels_np = utils.evaluate(image_processor, model, dataset, args.num_images, args.save, save_dir)
    
    utils.calibrate_and_plot(logits_np, labels_np, args.save, save_dir)
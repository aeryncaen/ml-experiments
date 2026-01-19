#!/usr/bin/env python3
"""Generate Path-X dataset using drewlinsley/pathfinder generator.

Usage:
    python scripts/generate_pathx.py --n-train 160000 --n-test 20000

Output: .cache/pathx/train/ and .cache/pathx/test/ with images and metadata.
"""
import argparse
import os
import sys

# Add pathfinder generator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".cache", "pathfinder_gen"))

import snakes2


class Args:
    """Configuration for pathfinder generation."""
    
    def __init__(
        self,
        contour_path="./contour",
        batch_id=0,
        n_images=10000,
        window_size=None,
        padding=22,
        antialias_scale=2,
        seed_distance=27,
        marker_radius=5,
        contour_length=14,
        distractor_length=None,
        num_distractor_snakes=None,
        snake_contrast_list=None,
        use_single_paddles=False,
        max_target_contour_retrial=4,
        max_distractor_contour_retrial=4,
        max_paddle_retrial=2,
        continuity=1.8,
        paddle_length=5,
        paddle_thickness=2,
        paddle_margin_list=None,
        paddle_contrast_list=None,
        pause_display=False,
        save_images=True,
        save_metadata=True,
        segmentation_task=False,
        segmentation_task_double_circle=False,
    ):
        self.contour_path = contour_path
        self.batch_id = batch_id
        self.n_images = n_images
        
        self.window_size = window_size or [256, 256]
        self.padding = padding
        self.antialias_scale = antialias_scale
        
        self.seed_distance = seed_distance
        self.marker_radius = marker_radius
        self.contour_length = contour_length
        
        # Derived from contour_length if not specified
        self.distractor_length = distractor_length or int(contour_length / 3)
        self.num_distractor_snakes = num_distractor_snakes or int(30 / self.distractor_length)
        
        self.snake_contrast_list = snake_contrast_list or [1.0]
        self.use_single_paddles = use_single_paddles
        
        self.max_target_contour_retrial = max_target_contour_retrial
        self.max_distractor_contour_retrial = max_distractor_contour_retrial
        self.max_paddle_retrial = max_paddle_retrial
        
        self.continuity = continuity
        self.paddle_length = paddle_length
        self.paddle_thickness = paddle_thickness
        self.paddle_margin_list = paddle_margin_list or [3]
        self.paddle_contrast_list = paddle_contrast_list or [1.0]
        
        self.pause_display = pause_display
        self.save_images = save_images
        self.save_metadata = save_metadata
        self.segmentation_task = segmentation_task
        self.segmentation_task_double_circle = segmentation_task_double_circle


def generate_split(output_dir: str, n_images: int, num_batches: int = 1, **kwargs):
    """Generate a dataset split."""
    os.makedirs(output_dir, exist_ok=True)
    
    images_per_batch = n_images // num_batches
    
    for batch_id in range(num_batches):
        print(f"\n=== Batch {batch_id + 1}/{num_batches} ===")
        args = Args(
            contour_path=output_dir,
            batch_id=batch_id,
            n_images=images_per_batch,
            **kwargs,
        )
        snakes2.from_wrapper(args)
    
    print(f"\nGenerated {n_images} images in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate Path-X dataset")
    parser.add_argument("--output-dir", type=str, default=".cache/pathx",
                        help="Output directory")
    parser.add_argument("--n-train", type=int, default=160000,
                        help="Number of training images")
    parser.add_argument("--n-val", type=int, default=20000,
                        help="Number of validation images")
    parser.add_argument("--n-test", type=int, default=20000,
                        help="Number of test images")
    parser.add_argument("--train-batches", type=int, default=16,
                        help="Number of batches for training set")
    parser.add_argument("--val-batches", type=int, default=2,
                        help="Number of batches for validation set")
    parser.add_argument("--test-batches", type=int, default=2,
                        help="Number of batches for test set")
    parser.add_argument("--size", type=int, default=256,
                        help="Image size (square)")
    parser.add_argument("--contour-length", type=int, default=14,
                        help="Length of target contours")
    parser.add_argument("--marker-radius", type=int, default=5,
                        help="Radius of endpoint markers")
    parser.add_argument("--continuity", type=float, default=1.8,
                        help="Snake continuity parameter")
    args = parser.parse_args()
    
    # Common generation kwargs
    gen_kwargs = dict(
        window_size=[args.size, args.size],
        contour_length=args.contour_length,
        marker_radius=args.marker_radius,
        continuity=args.continuity,
    )
    
    # Generate training set
    if args.n_train > 0:
        print(f"\n{'='*60}")
        print(f"Generating TRAINING set: {args.n_train} images")
        print(f"{'='*60}")
        generate_split(
            os.path.join(args.output_dir, "train"),
            args.n_train,
            num_batches=args.train_batches,
            **gen_kwargs,
        )
    
    # Generate validation set
    if args.n_val > 0:
        print(f"\n{'='*60}")
        print(f"Generating VALIDATION set: {args.n_val} images")
        print(f"{'='*60}")
        generate_split(
            os.path.join(args.output_dir, "val"),
            args.n_val,
            num_batches=args.val_batches,
            **gen_kwargs,
        )
    
    # Generate test set
    if args.n_test > 0:
        print(f"\n{'='*60}")
        print(f"Generating TEST set: {args.n_test} images")
        print(f"{'='*60}")
        generate_split(
            os.path.join(args.output_dir, "test"),
            args.n_test,
            num_batches=args.test_batches,
            **gen_kwargs,
        )
    
    print(f"\n{'='*60}")
    print("Dataset generation complete!")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

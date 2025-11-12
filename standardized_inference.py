"""
Standardized inference script for TMF-Matting
Structure: preprocess -> load_model -> infer -> post_process
"""
import os
import argparse


# ============================================================================
# 1. PREPROCESSING
# ============================================================================
def preprocess(image_paths, trimap_paths, model, device='cuda:0'):
    """
    Preprocess input images and trimaps for batch inference.

    Args:
        image_paths: List of image file paths
        trimap_paths: List of trimap file paths
        model: The loaded model (needed for config pipeline)
        device: Device to load tensors to

    Returns:
        batched_data: List of preprocessed data dictionaries
    """
    from mmedit.datasets.pipelines import Compose

    cfg = model.cfg

    # Remove alpha from test_pipeline (for inference)
    keys_to_remove = ['alpha', 'ori_alpha']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                if pipeline in cfg.test_pipeline:
                    cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0 and pipeline in cfg.test_pipeline:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)

    # Build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    # Preprocess each image-trimap pair
    batched_data = []
    for img_path, trimap_path in zip(image_paths, trimap_paths):
        data = dict(merged_path=img_path, trimap_path=trimap_path)
        data = test_pipeline(data)
        batched_data.append(data)

    return batched_data


# ============================================================================
# 2. MODEL LOADING
# ============================================================================
def load_model(config_path, checkpoint_path=None, device='cuda:0'):
    """
    Load the TMF-Matting model with pretrained weights.

    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        model: Loaded model in eval mode
    """
    import mmcv
    from mmcv.runner import load_checkpoint
    from mmedit.models import build_model

    # Load configuration
    if isinstance(config_path, str):
        config = mmcv.Config.fromfile(config_path)
    elif not isinstance(config_path, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config_path)}')

    # Prepare config
    config.model.pretrained = None
    config.test_cfg.metrics = None

    # Build model
    model = build_model(config.model, test_cfg=config.test_cfg)

    # Load checkpoint
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(model, checkpoint_path)

    # Save config and set to eval mode
    model.cfg = config
    model.to(device)
    model.eval()

    return model


# ============================================================================
# 3. INFERENCE
# ============================================================================
def infer(model, batched_data, device='cuda:0'):
    """
    Run inference on batched preprocessed data.

    Args:
        model: Loaded model
        batched_data: List of preprocessed data dictionaries
        device: Device for inference

    Returns:
        outputs: List of model outputs (predicted alpha mattes)
    """
    import torch
    from mmcv.parallel import collate, scatter

    outputs = []

    with torch.no_grad():
        for data in batched_data:
            # Scatter data to device
            data_gpu = scatter(collate([data], samples_per_gpu=1), [device])[0]

            # Forward pass
            result = model(test_mode=True, **data_gpu)

            outputs.append(result)

    return outputs


# ============================================================================
# 4. POST-PROCESSING
# ============================================================================
def post_process(outputs, output_dir='./output', save_names=None):
    """
    Post-process model outputs to create final alpha mattes.

    Args:
        outputs: List of model output dictionaries
        output_dir: Directory to save output images
        save_names: List of output filenames (without extension)

    Returns:
        alphas: List of alpha matte numpy arrays (values in [0, 1])
    """
    import numpy as np
    import mmcv

    os.makedirs(output_dir, exist_ok=True)

    alphas = []

    for i, result in enumerate(outputs):
        # Extract predicted alpha
        alpha = result['pred_alpha']

        # Alpha is already in [0, 1] range and numpy format
        alpha = np.clip(alpha, 0, 1)

        alphas.append(alpha)

        # Save if output directory and names provided
        if save_names is not None and i < len(save_names):
            # Save as grayscale PNG (0-255)
            alpha_uint8 = (alpha * 255).astype(np.uint8)
            save_path = os.path.join(output_dir, f'{save_names[i]}_alpha.png')
            mmcv.imwrite(alpha_uint8, save_path)

    return alphas


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Standardized inference script for TMF-Matting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/mattors/gca/gca_r34_4x10_200k_comp1k.py',
        help='Path to model config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./work_dirs/gca_r34_4x10_200k_comp1k/latest.pth',
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        default=['./demo/sample_image_1.png', './demo/sample_image_2.png'],
        help='List of input image paths'
    )
    parser.add_argument(
        '--trimaps',
        type=str,
        nargs='+',
        default=['./demo/sample_trimap_1.png', './demo/sample_trimap_2.png'],
        help='List of input trimap paths'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory to save output alpha mattes'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use for inference (e.g., cuda:0 or cpu)'
    )
    return parser.parse_args()


def main():
    """
    Main inference pipeline with command-line argument support.
    """
    import torch

    # Parse arguments
    args = parse_args()

    # Determine device
    if 'cuda' in args.device and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device

    config_path = args.config
    checkpoint_path = args.checkpoint
    image_paths = args.images
    trimap_paths = args.trimaps
    output_dir = args.output_dir

    # Validate inputs
    if len(image_paths) != len(trimap_paths):
        raise ValueError(f"Number of images ({len(image_paths)}) must match number of trimaps ({len(trimap_paths)})")

    # Generate save names from image filenames
    save_names = [os.path.splitext(os.path.basename(img))[0] for img in image_paths]

    print("=" * 80)
    print("TMF-MATTING STANDARDIZED INFERENCE")
    print("=" * 80)

    # Step 1: Load Model (must be done before preprocessing in TMF-Matting)
    print("\n[1/4] Loading model...")
    model = load_model(config_path, checkpoint_path, device=device)
    print(f"  - Config: {config_path}")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Device: {device}")

    # Step 2: Preprocess
    print("\n[2/4] Preprocessing images...")
    batched_data = preprocess(image_paths, trimap_paths, model, device=device)
    print(f"  - Batch size: {len(batched_data)}")

    # Step 3: Inference
    print("\n[3/4] Running inference...")
    outputs = infer(model, batched_data, device=device)
    print(f"  - Processed {len(outputs)} images")

    # Step 4: Post-process
    print("\n[4/4] Post-processing outputs...")
    alphas = post_process(outputs, output_dir=output_dir, save_names=save_names)
    print(f"  - Generated {len(alphas)} alpha mattes")
    print(f"  - Saved to: {output_dir}/")

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)

    return alphas


if __name__ == '__main__':
    main()

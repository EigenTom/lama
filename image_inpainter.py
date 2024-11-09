# predictor.py

import logging
import os
import sys
import traceback

import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

class Predictor:
    def __init__(self, prediction_cfg_path, model_path, checkpoint_name, device='cpu', refine=False):
        """
        Initializes the Predictor with model configuration and device setup.
        """
        # Set environment variables to limit CPU thread usage
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # Load model configuration
        with open(prediction_cfg_path, 'r') as f:
            self.predict_config = OmegaConf.create(yaml.safe_load(f))
        
        self.device = torch.device(device)
        self.refine = refine
        self.model_path = model_path
        self.checkpoint_name = checkpoint_name


        # Load the training configuration for the model
        train_config_path = os.path.join(model_path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            self.train_config = OmegaConf.create(yaml.safe_load(f))

        # Configure model for prediction only
        self.train_config.training_model.predict_only = True
        self.train_config.visualizer.kind = 'noop'

        # Load and prepare the model
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the model checkpoint and applies necessary configuration.
        """
        checkpoint_path = os.path.join(self.model_path, 'models', self.checkpoint_name)
        model = load_checkpoint(self.train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not self.refine:
            model.to(self.device)
        return model

    def predict(self, indir, outdir, out_ext='.png', out_key='output'):
        """
        Runs the prediction on each image in the input directory and saves the output.
        """
        # if sys.platform != 'win32':
        #     register_debug_signal_handlers()  # Signal handler for debug
        dataset = make_default_val_dataset(indir, **self.predict_config.dataset)
        
        if not indir.endswith('/'):
            indir += '/'
        
        for img_i in tqdm.trange(len(dataset)):
            # Setup output filename
            
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(outdir, os.path.splitext(mask_fname[len(indir):])[0] + out_ext)
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            
            # Collate and process batch
            batch = default_collate([dataset[img_i]])
            
            if self.predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for refinement."
                cur_res = refine_predict(batch, self.model, **self.predict_config.refiner)
                cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = self.model(batch)                    
                    cur_res = batch[self.predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]
            
            # Save result
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)
            LOGGER.info(f"Saved prediction to {cur_out_fname}")



# from predictor import Predictor

# Define model and input/output parameters
prediction_cfg_path = "/home/axton/axton-workspace/csc2125/models/lama/configs/prediction/default.yaml"
model_path = "/home/axton/axton-workspace/csc2125/model_weights/lama_weights"
checkpoint_name = "best.ckpt"
indir = "/home/axton/axton-workspace/csc2125/models/lama/imgs"
outdir = "/home/axton/axton-workspace/csc2125/models/lama/output"
device = "cuda"  # or "cuda" if GPU is available
refine = False  # Set True if refinement is needed

# Initialize predictor
predictor = Predictor(
    prediction_cfg_path=prediction_cfg_path,
    model_path=model_path,
    checkpoint_name=checkpoint_name,
    device=device,
    refine=refine
)

# Run prediction
predictor.predict(
    indir=indir,
    outdir=outdir,
    out_ext=".png",
    out_key="output"  # Specify output key if different from default
)

print(f"Predictions saved to {outdir}")

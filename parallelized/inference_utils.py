import torch
import os
import logging
import argparse
import ray
from abodybuilder3.utils import string_to_input, output_to_pdb, add_atom37_to_output
from abodybuilder3.lightning_module import LitABB3

# Set up logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_weights_directory: str, parallelize_with_ray: bool = False):
    try:
        # Load the model
        module = LitABB3.load_from_checkpoint(model_weights_directory)
        model = module.model

        # Set the model to evaluation mode
        model.eval()

        # Fix the seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Freeze the model weights (optional, but good for inference)
        for param in model.parameters():
            param.requires_grad = False

        if parallelize_with_ray:
            # If parallelization is enabled, use Ray's object store
            return ray.put(model)  # Put the full model in Ray's object store
        else:
            # Otherwise, return the model directly
            return model

    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def generate_pdb(heavy: str, light: str, name: str, model):
    try:
        # Set device (CUDA or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare input
        ab_input = string_to_input(heavy=heavy, light=light)
        ab_input_batch = {
            key: (value.unsqueeze(0).to(device) if key not in ["single", "pair"] else value.to(device))
            for key, value in ab_input.items()
        }

        # Move model to device
        model.to(device)

        # Run inference
        output = model(ab_input_batch, ab_input_batch["aatype"])
        output = add_atom37_to_output(output, ab_input["aatype"].to(device))

        # Generate PDB string
        pdb_string = output_to_pdb(output, ab_input)

        # Return the PDB string instead of saving to a file
        return pdb_string

    except Exception as e:
        # Log the error along with sequences if something goes wrong
        logging.error(f"Error processing sequences:\nName: {name}\nHeavy Chain: {heavy}\nLight Chain: {light}\nError: {str(e)}")
        return None  # Gracefully handle errors

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate PDB from antibody sequences using ABodyBuilder.")
    parser.add_argument("--heavy_seq", required=True, help="Heavy chain sequence of the antibody.")
    parser.add_argument("--light_seq", required=True, help="Light chain sequence of the antibody.")
    parser.add_argument("--model_weights_dir", required=True, help="Path to the model weights directory.")
    parser.add_argument("--antibody_num", required=True, type=int, help="Antibody number to be used for PDB filename.")
    
    return parser.parse_args()

# Main entry point
if __name__ == "__main__":
    args = parse_arguments()

    try:
        # First, load the model
        model = load_model(args.model_weights_dir)

        # Then, generate the PDB string
        pdb_string = generate_pdb(heavy=args.heavy_seq, light=args.light_seq, 
                                  model=model, 
                                  antibody_num=args.antibody_num)
        
        if pdb_string:
            print(f"PDB string generated successfully. First 100 characters:\n{pdb_string[:100]}...")
        else:
            print("Failed to generate PDB.")
        
    except Exception as e:
        print(f"An error occurred. Please check the error log for details.")

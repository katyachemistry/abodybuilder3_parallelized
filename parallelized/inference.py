import os
import logging
import pandas as pd
import ray
import argparse
from inference_utils import generate_pdb, load_model
import time

def main(csv_filename, output_directory, num_gpus):
    # Load the data
    abs = pd.read_csv(csv_filename, usecols=['Therapeutic', 'HeavySequence', 'LightSequence'])
    sequences = abs[['HeavySequence', 'LightSequence']].values
    names = abs['Therapeutic']

    # Define paths
    path_to_model = './output/plddt-loss/best_second_stage.ckpt'

    # Initialize Ray
    ray.shutdown()  # Reset the ray environment in case it was already running
    ray.init(ignore_reinit_error=True)

    # Load model
    model_ref = load_model(path_to_model, parallelize_with_ray=True)
    model = ray.get(model_ref)

    # Configure logging
    log_file = "error_log_prediction.txt"
    logging.basicConfig(filename=log_file, level=logging.ERROR, 
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # Define the remote function with dynamic GPU allocation
    @ray.remote(num_gpus=num_gpus)  # Use the num_gpus parameter here
    def predict_structure(heavy, light, name, model, out_path, write_pdb=True):
        """Predicts the protein structure and writes it to a PDB file if needed.
        
        - Skips processing if the file already exists.
        - Logs errors to a file instead of just printing.
        """
        try:
            # Ensure the output directory exists
            os.makedirs(out_path, exist_ok=True)

            output_file = os.path.join(out_path, name + '.pdb')

            # Check if file already exists to avoid redundant work
            if os.path.exists(output_file):
                return None

            # Run the structure prediction
            pdb = generate_pdb(heavy=heavy, light=light, model=model, name=name)

            if write_pdb:
                try:
                    with open(output_file, 'w') as file:
                        file.write(pdb)
                except IOError as e:
                    error_msg = f"Error writing file {output_file}: {e}"
                    logging.error(error_msg)
                return None

            return pdb

        except Exception as e:
            logging.error(f"Error predicting structure for {name}: {e}")
            return None

    # Launch the tasks
    tasks = []
    for name, (heavy, light) in zip(names, sequences):
        tasks.append(predict_structure.remote(heavy, light, name, model, output_directory, write_pdb=True))

    # Wait for tasks to complete as they finish, rather than waiting for all at once
    while tasks:
        # Wait for at least one task to complete
        done, tasks = ray.wait(tasks, num_returns=1, timeout=1000)  # You can adjust the timeout as needed
        
        # Optionally process the done task here if needed (e.g., logging, etc.)
        # You could also use ray.get(done) if you need to gather results immediately

    ray.shutdown()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Predict protein structures and save to PDB files.")
    parser.add_argument('csv_filename', type=str, help="Path to the input CSV file.")
    parser.add_argument('output_directory', type=str, help="Directory where the PDB files will be saved.")
    parser.add_argument('--num_gpus', type=float, default=1, help="Number of GPUs to allocate per task (default is 1).")

    args = parser.parse_args()

    # Call the main function with command-line arguments
    start = time.time()
    main(args.csv_filename, args.output_directory, args.num_gpus)
    print(f'Inference took {time.time() - start} seconds')

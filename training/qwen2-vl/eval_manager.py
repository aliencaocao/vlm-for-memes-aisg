# Usage example: python eval_manager.py --ckpt checkpoints/qwen2-vl-7b-pissa-r128-a256-lr1e4-without-sg-no-eval-sampling

import argparse
import glob
import logging
import os
import subprocess
import tempfile
import time
from multiprocessing import Lock, Process, Manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open('a100_inference.yaml') as f:
    a100_config = f.read().splitlines()
with open('inference.yaml') as f:
    v100_config = f.read().splitlines()


def mod_ckpt(config, ckpt):
    for i, line in enumerate(config):
        if line.startswith('adapter_name_or_path'):
            config[i] = f'adapter_name_or_path: {ckpt}'
        elif line.startswith('run_name'):
            config[i] = f'run_name: {"/".join(ckpt.split("/")[1:])}'
    return config


# Queue to hold the jobs
job_queue = Manager().Queue()

# List of available GPUs
available_gpus = Manager().Queue(maxsize=5)  # 0 to 4
for i in range(5):
    available_gpus.put(i)

# Lock for thread-safe operations
gpu_lock = Lock()


def run_job(gpu_id, ckpt):
    # Create a temporary file for the modified YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        new_config = mod_ckpt(a100_config if gpu_id == 0 else v100_config, ckpt)
        temp_file.write("\n".join(new_config))
        temp_file_path = temp_file.name

    # Prepare the command to run
    cmd = f"""
    ulimit -n 1048576 && \
    CUDA_VISIBLE_DEVICES={gpu_id} WANDB_USERNAME='aisg-meme' WANDB_PROJECT='qwen2-vl' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DISABLE_VERSION_CHECK=True llamafactory-cli train {temp_file_path}
    """

    # # for debugging
    # cmd = f"""
    # echo "Running {ckpt} on GPU{gpu_id}" && \
    # sleep 10
    # """

    # Start the job but do not wait for it to finish
    logging.info(f"Started job: {ckpt} on GPU{gpu_id}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        logging.error(f"Job failed: {ckpt} on GPU{gpu_id}")
    else:
        logging.info(f"Job completed: {ckpt} on GPU{gpu_id}")

    # Remove the temporary file
    os.remove(temp_file_path)

    # Mark the GPU as available
    with gpu_lock:
        available_gpus.put(gpu_id)

    logging.info(f"GPU{gpu_id} is now available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Folder containing the various checkpoints of a run')
    args = parser.parse_args()

    ckpt_root = args.ckpt
    checkpoints = sorted(glob.glob(f'{ckpt_root}/*/checkpoint-*') + glob.glob(f'{ckpt_root}/checkpoint-*'))
    logging.info(f'Found {len(checkpoints)} checkpoints:\n{checkpoints}')

    # Load all jobs into queue
    for ckpt in checkpoints:
        job_queue.put(ckpt)

    gpu_procs = []
    while not job_queue.empty():
        if not available_gpus.empty():
            with gpu_lock:
                gpu_id = available_gpus.get()
                ckpt = job_queue.get()
                logging.info(f"Enqueuing {ckpt} on GPU{gpu_id}")
                proc = Process(target=run_job, args=(gpu_id, ckpt))
                proc.start()
                gpu_procs.append(proc)
        time.sleep(5)  # check for free gpu every 5 seconds

    # Wait for all jobs to complete
    for proc in gpu_procs:
        proc.join()

    logging.info("All jobs completed!")


if __name__ == "__main__":
    main()

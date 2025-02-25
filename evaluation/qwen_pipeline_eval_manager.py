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

# Queue to hold the jobs
job_queue = Manager().Queue()

# List of available GPUs
available_gpus = Manager().Queue(maxsize=5)  # 0 to 4
for i in range(5):
    available_gpus.put(i)

# Lock for thread-safe operations
gpu_lock = Lock()


def run_job(gpu_id, ckpt):
    # Prepare the command to run
    cmd = f"""
    MODEL_PATH={ckpt} CUDA_VISIBLE_DEVICES={gpu_id} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python qwen_pipeline_eval.py
    """

    # Start the job but do not wait for it to finish
    logging.info(f"Started job: {ckpt} on GPU{gpu_id}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        logging.error(f"Job failed: {ckpt} on GPU{gpu_id}")
    else:
        logging.info(f"Job completed: {ckpt} on GPU{gpu_id}")

    # Mark the GPU as available
    with gpu_lock:
        available_gpus.put(gpu_id)

    logging.info(f"GPU{gpu_id} is now available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Folder containing the various merged models')
    args = parser.parse_args()

    ckpt_root = args.ckpt
    checkpoints = sorted(glob.glob(f'{ckpt_root}/*'))
    logging.info(f'Found {len(checkpoints)} models:\n{checkpoints}')

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

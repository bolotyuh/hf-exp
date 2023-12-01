import concurrent.futures
from tqdm import tqdm
import subprocess
from pathlib import Path
import os


def run_command(file):
    cmd = f"sudo lfs hsm_restore {file}"
    result = subprocess.run(cmd, shell=True)
    return True

def main():
    files = Path('/fsx/data/train').glob('**/*.jpg')
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(run_command, files), total=len(files), desc="Processing"))
        
if __name__ == "__main__":
    main()
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
    dirs = sorted(list(Path('/fsx/data/train').iterdir()))
    files = []
    for i in tqdm(range(1201, 4000)):
        files.extend(list(Path(dirs[i]).glob('*.jpg')))
    print(len(files))
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count()*4)) as executor:
        list(tqdm(executor.map(run_command, files), total=len(files), desc="Processing"))

if __name__ == "__main__":
    main()
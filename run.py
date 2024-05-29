import subprocess
import concurrent.futures
import time


def run_script(script, name, **kwargs):
    cmd = ["python", script, "--name", str(name)]
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error running {script}")
        return False
    return True


def run_main_scripts(satellite, name):
    # Define the scripts to be run concurrently and sequentially
    concurrent_scripts = ["main_RSP.py", "main_SDE.py"]
    sequential_scripts = ["main_FUG.py", "test.py"]

    # Parameters for main_RSP.py
    rsp_params = {
        "lr": 0.001,
        "epochs": 80,
        "batch_size": 8,
        "device": 'cuda',
        "satellite": satellite,
        "name": name
    }

    # Parameters for main_SDE.py
    sde_params = {
        "lr": 0.0005,
        "epochs": 250,
        "batch_size": 1,
        "device": 'cuda',
        "satellite": satellite,
        "name": name
    }

    # Parameters for main_FUG.py
    fug_params = {
        "lr": 0.0005,
        "epochs": 50,
        "batch_size": 1,
        "device": 'cuda',
        "satellite": satellite,
        "name": name
    }

    # Parameters for test.py
    test_params = {
        "satellite": satellite,
        "name": name
    }

    # Run concurrent scripts using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_script, "main_RSP.py", **rsp_params),
            executor.submit(run_script, "main_SDE.py", **sde_params)
        ]

        # Wait for concurrent scripts to complete
        concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # If any concurrent script failed, return False
        if not all(concurrent_results):
            return False

    # Run sequential scripts
    for script in sequential_scripts:
        if script == "main_FUG.py":
            if not run_script(script, **fug_params):
                return False
        elif script == "test.py":
            if not run_script(script, **test_params):
                return False
        else:
            if not run_script(script, name):
                return False

    return True


if __name__ == "__main__":
    current_satellite = 'wv3/'
    current_name = 19
    print(f'training data is {current_satellite}{current_name}')
    t1 = time.time()
    if run_main_scripts(current_satellite, current_name):
        print(f"Completed run with NAME={current_name}")
    else:
        print("Script execution failed. Exiting...")
    t2 = time.time()
    print(f'total time: {t2-t1}s')

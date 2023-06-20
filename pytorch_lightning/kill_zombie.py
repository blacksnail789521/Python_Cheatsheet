import os
import psutil
import signal


def kill_processes_by_name_and_status(target_name, target_status):
    for process in psutil.process_iter(["pid", "name", "status"]):
        if (
            process.info["status"] == target_status
            and process.info["name"] == target_name
        ):
            print(
                f"Killing process: {process.info['name']} (PID: {process.info['pid']}) with status {target_status}"
            )
            os.kill(process.info["pid"], signal.SIGTERM)


if __name__ == "__main__":
    process_name = "ray::ImplicitFunc.train"
    process_status = psutil.STATUS_ZOMBIE  # or any other valid status
    kill_processes_by_name_and_status(process_name, process_status)

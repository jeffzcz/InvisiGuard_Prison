import subprocess
import time
import os
import logging
import sys

# --- Configuration ---
# This is the root directory where the launcher is.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the new directory and file structure
VERSIONS_DIR = os.path.join(BASE_DIR, "versions")
SHARED_DATA_DIR = os.path.join(BASE_DIR, "shared_data")

# --- Key files for version and update management ---
# This file tells the launcher which version to run.
CURRENT_VERSION_FILE = os.path.join(BASE_DIR, "current_version.txt")
# This file is a backup of the last known good version during an update.
PREVIOUS_VERSION_FILE = os.path.join(BASE_DIR, "previous_version.txt")

# --- Flag files for communication between main_code.py and launcher.py ---
# The main app creates this file to signal an update is ready.
UPDATE_FLAG = os.path.join(SHARED_DATA_DIR, "update.flag")
# The main app creates this file to signal it has started successfully.
HEALTH_PING_FILE = os.path.join(SHARED_DATA_DIR, "health.ping")
# The launcher writes to this file if a rollback occurs.
ROLLBACK_LOG_FILE = os.path.join(SHARED_DATA_DIR, "rollback.log")

# --- Launcher Settings ---
# Time to wait for the new version to create health.ping before rolling back.
HEALTH_CHECK_TIMEOUT_SECONDS = 90
# Time to wait after a crash before restarting (prevents fast crash loops).
CRASH_RESTART_DELAY_SECONDS = 10

# --- Logging ---
# We'll log launcher actions to its own file in the shared_data directory.
LOG_FILE = os.path.join(SHARED_DATA_DIR, "launcher.log")

# Ensure shared_data directory exists
os.makedirs(SHARED_DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Launcher] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)  # Also print to console
    ]
)
logger = logging.getLogger(__name__)


def read_file(file_path):
    """Safely read the content of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def write_file(file_path, content):
    """Safely write content to a file."""
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return False

def delete_file(file_path):
    """Safely delete a file if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")

def get_current_version():
    """Get the version we are supposed to run from current_version.txt."""
    version = read_file(CURRENT_VERSION_FILE)
    if not version:
        logger.error(f"CRITICAL: Cannot read {CURRENT_VERSION_FILE}. Cannot start.")
        sys.exit(1)  # Exit if we don't know what to run
    return version

def start_worker_process(version_to_run):
    """Starts the main_code.py for a specific version."""
    version_dir = os.path.join(VERSIONS_DIR, version_to_run)
    script_path = os.path.join(version_dir, "main_code.py")

    if not os.path.exists(script_path):
        logger.error(f"Cannot find main_code.py for version {version_to_run} at {script_path}")
        return None

    logger.info(f"Starting worker process: version {version_to_run}")
    # We pass the shared_data_dir and version as arguments to main_code.py
    # This makes main_code.py independent of the directory structure.
    try:
        process = subprocess.Popen(
            [sys.executable, script_path, "--shared_data_dir", SHARED_DATA_DIR, "--version", version_to_run],
            cwd=version_dir  # Set the working directory to the version folder
        )
        return process
    except Exception as e:
        logger.error(f"Failed to start process for version {version_to_run}: {e}")
        return None

def perform_rollback(failed_version):
    """Rolls back to the previous known good version."""
    logger.warning(f"--- ROLLBACK INITIATED for {failed_version} ---")
    
    # Get the version to roll back to
    rollback_version = read_file(PREVIOUS_VERSION_FILE)
    
    if not rollback_version:
        logger.error(f"CRITICAL: Rollback failed. No {PREVIOUS_VERSION_FILE} found.")
        # At this point, we are stuck. We'll just try the failed version again.
        # A more robust system might enter a "safe mode".
        write_file(ROLLBACK_LOG_FILE, f"Rollback failed for {failed_version}: No previous_version.txt found.")
        return failed_version # Return the version we just tried

    # Log the rollback
    log_message = f"Health check failed for {failed_version}. Rolling back to {rollback_version}."
    logger.warning(log_message)
    write_file(ROLLBACK_LOG_FILE, log_message)
    
    # Update current_version.txt to the old, good version
    write_file(CURRENT_VERSION_FILE, rollback_version)
    
    # Clean up the previous version file
    delete_file(PREVIOUS_VERSION_FILE)
    
    logger.info(f"Rollback complete. Will restart with {rollback_version}.")
    return rollback_version

def perform_update():
    """Handles the update process by reading flags and swapping versions."""
    logger.info("Update flag detected. Starting update process...")
    
    new_version = read_file(UPDATE_FLAG)
    if not new_version:
        logger.error("Update flag found, but file is empty. Aborting update.")
        delete_file(UPDATE_FLAG)
        return None # No version change

    current_version = get_current_version()
    
    if new_version == current_version:
        logger.warning(f"Update flag points to same version ({new_version}). Ignoring.")
        delete_file(UPDATE_FLAG)
        return None # No version change

    logger.info(f"Updating from {current_version} to {new_version}")

    # 1. Backup: Save the current (known good) version
    write_file(PREVIOUS_VERSION_FILE, current_version)
    
    # 2. Update: Set the new version as current
    write_file(CURRENT_VERSION_FILE, new_version)
    
    # 3. Clean up the update flag
    delete_file(UPDATE_FLAG)
    
    return new_version # Return the new version to be launched

def health_check(process, version_being_checked):
    """
    Monitors for a health.ping file to confirm the new version is healthy.
    Returns True if healthy, False if rollback is needed.
    """
    logger.info(f"Starting health check for {version_being_checked} (Timeout: {HEALTH_CHECK_TIMEOUT_SECONDS}s)")
    
    # Clear any old health ping
    delete_file(HEALTH_PING_FILE)
    
    start_time = time.time()
    while time.time() - start_time < HEALTH_CHECK_TIMEOUT_SECONDS:
        # Check if the process crashed during health check
        if process.poll() is not None:
            logger.warning(f"Process {version_being_checked} crashed during health check (exit code {process.poll()}).")
            return False # Trigger rollback
        
        # Check if the health ping file appeared
        if os.path.exists(HEALTH_PING_FILE):
            logger.info(f"Health check PASSED for {version_being_checked}.")
            # Commit the update by removing the rollback file
            delete_file(PREVIOUS_VERSION_FILE)
            delete_file(HEALTH_PING_FILE) # Clean up
            return True # Healthy

        time.sleep(1) # Poll every second
    
    # If we get here, the timer expired
    logger.error(f"Health check FAILED for {version_being_checked}. Timeout expired.")
    
    # Terminate the unhealthy process
    try:
        process.terminate()
        process.wait(timeout=5)
    except:
        process.kill()
        
    return False # Trigger rollback

def main_loop():
    """The main watchdog and launcher loop."""
    
    current_version = get_current_version()
    worker_process = start_worker_process(current_version)
    
    while True:
        if worker_process is None:
            logger.error(f"Worker process {current_version} failed to start. Retrying in {CRASH_RESTART_DELAY_SECONDS}s.")
            time.sleep(CRASH_RESTART_DELAY_SECONDS)
            worker_process = start_worker_process(current_version)
            continue

        # Wait for the process to exit
        exit_code = worker_process.poll()

        if exit_code is None:
            # Process is still running, sleep and check again
            time.sleep(2)
            continue
            
        logger.warning(f"Worker process {current_version} has exited with code {exit_code}.")
        
        # --- Process has exited, decide what to do ---

        # 1. Check for an update signal
        if os.path.exists(UPDATE_FLAG):
            new_version = perform_update()
            if new_version:
                # Update was successful, launch new version
                current_version = new_version
                worker_process = start_worker_process(current_version)
                
                # Start health check for the new version
                if not health_check(worker_process, current_version):
                    # Health check failed, perform rollback
                    current_version = perform_rollback(current_version)
                    # The loop will now restart with the rolled-back version
                    worker_process = None # Force restart
                    continue 
            else:
                # Update failed (e.g., flag was empty), restart same version
                logger.warning("Update failed. Restarting same version.")
                worker_process = start_worker_process(current_version)
        
        # 2. No update flag, so it was a crash or normal exit
        else:
            logger.warning(f"Process {current_version} exited unexpectedly (crash?).")
            logger.info(f"Restarting version {current_version} in {CRASH_RESTART_DELAY_SECONDS}s.")
            time.sleep(CRASH_RESTART_DELAY_SECONDS)
            worker_process = start_worker_process(current_version)

if __name__ == "__main__":
    logger.info("--- InvisiGuard Launcher Started ---")
    logger.info(f"Base Directory: {BASE_DIR}")
    logger.info(f"Shared Data Directory: {SHARED_DATA_DIR}")
    
    # Check for a previous_version.txt file at startup
    # This implies a hard crash (e.g., power loss) during an update.
    if os.path.exists(PREVIOUS_VERSION_FILE):
        logger.warning("Found 'previous_version.txt' on startup.")
        perform_rollback(get_current_version()) # Roll back to be safe
    
    main_loop()

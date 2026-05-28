# InvisiGuard Prison Demo

InvisiGuard is a smart monitoring and tracking system that utilizes multiple sensors—including Novelda and TI radars—alongside YOLO skeleton tracking to monitor health metrics (BPM, distance) and track individuals. This repository contains the core application along with a robust launcher system designed for high reliability and seamless updates.

## System Architecture

The application is structured to ensure maximum uptime, providing crash recovery and safe version updates:

- **`launcher.py`**: The primary entry point. It monitors the worker process, starts the correct version, handles crashes, and performs updates with automatic rollbacks if the new version fails a health check.
- **`current_version.txt`**: Indicates the currently active version of the application.
- **`versions/`**: Contains the codebase for specific versions (e.g., `versions/1.1.0/main_code.py`). This allows multiple versions to coexist for easy rollbacks.
- **`shared_data/`**: Stores persistent data that survives version updates, such as radar logs, device configurations (`device_config.ini`), security certificates, and system flags (`update.flag`, `health.ping`).

## Key Features
- **Radar Data Processing**: Integration with Novelda and TI mmWave radars for presence and vital sign monitoring.
- **Health Monitoring**: Real-time extraction of heartbeat (BPM) and subject distance.
- **YOLO Skeleton Tracking**: Advanced visual tracking using `yolo_skeleton.py`.
- **Auto-Recovery**: The launcher detects application crashes and automatically restarts the process to prevent downtime.
- **Safe Updates**: Hot-swapping to new versions via the `update.flag`. If the new version fails to ping its health within the timeout, the launcher automatically rolls back to the previous version.

## How to Run

To start the system, simply run the launcher. Do not run the worker scripts directly.

```bash
python launcher.py
```

The launcher will read `current_version.txt`, start the corresponding worker process (e.g., `versions/1.1.0/main_code.py`), and continuously monitor its health.
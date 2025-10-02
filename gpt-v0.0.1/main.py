#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Orchestrates launching the Flower Federated Learning server and four clients.
- Starts server.py first, waits briefly, then launches 4 client.py instances with unique client IDs 0..3. [web:24][web:46]
- Uses subprocess to run processes concurrently and ensures cleanup with a try-finally block. [web:39][web:38]
"""

import subprocess
import sys
import time
import os
from typing import List

def launch_process(cmd: List[str], env=None) -> subprocess.Popen:
    """
    Launch a subprocess with the given command list, returning the Popen object. [web:39]
    """
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )  # [web:39]

def terminate_processes(procs: List[subprocess.Popen]) -> None:
    """
    Gracefully terminate all processes, then force-kill any that remain. [web:39][web:45]
    """
    for p in procs:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass
    # Give processes a moment to exit gracefully
    time.sleep(1.0)
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass  # [web:39][web:45]

def stream_output(name: str, proc: subprocess.Popen) -> None:
    """
    Non-blocking read helpers could be implemented, but to keep it simple and cross-platform,
    this function is unused. Output is still captured for debugging. [web:39]
    """
    _ = name
    _ = proc  # [web:39]

def main():
    server_cmd = [sys.executable, "server.py", "--server_address", "[::]:8080", "--num_rounds", "10"]
    client_base_cmd = [sys.executable, "client.py", "--server_address", "0.0.0.0:8080"]  # matches server bind. [web:46]

    processes: List[subprocess.Popen] = []
    try:
        # 1) Start server first
        print("Starting Flower server...", flush=True)  # [web:24]
        server_proc = launch_process(server_cmd)
        processes.append(server_proc)

        # Wait briefly to ensure the server is ready
        time.sleep(3.0)  # adjust if needed based on environment startup time. [web:24]

        # 2) Start four clients with unique client_id 0..3
        for cid in range(4):
            cmd = client_base_cmd + ["--client_id", str(cid)]
            print(f"Starting client {cid}...", flush=True)  # [web:46]
            p = launch_process(cmd)
            processes.append(p)

        # 3) Wait for the server to finish (which ends after configured rounds)
        # Clients will typically exit once the server completes training. [web:24]
        server_return = server_proc.wait()
        print(f"Server exited with code {server_return}", flush=True)  # [web:24]

        # Optionally, wait for clients to exit; add time limit to avoid hanging
        end_time = time.time() + 30.0
        for p in processes[1:]:
            timeout = max(0.0, end_time - time.time())
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass  # [web:38][web:39]

    finally:
        # 4) Cleanup: ensure all processes are terminated
        terminate_processes(processes)  # [web:39][web:45]
        print("All subprocesses terminated.", flush=True)

if __name__ == "__main__":
    main()  # [web:24][web:39]

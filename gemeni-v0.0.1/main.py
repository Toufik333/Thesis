# main.py

import subprocess
import sys
import time

# --- Configuration ---
NUM_CLIENTS = 4
SERVER_SCRIPT = "server.py"
CLIENT_SCRIPT = "client.py"

def main():
    """
    Orchestrates the start of the Flower server and multiple clients.
    
    This script starts the server first, waits for it to initialize, and then
    launches the specified number of client processes. It ensures all child
    processes are terminated cleanly when the script exits.
    """
    processes = []
    
    # Use the same Python interpreter that is running this script
    python_executable = sys.executable

    try:
        # 1. Start the Flower server
        print("🚀 Starting Flower Server...")
        server_command = [python_executable, SERVER_SCRIPT]
        server_process = subprocess.Popen(
            server_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(server_process)
        print(f"✅ Server started with PID: {server_process.pid}")
        
        # Give the server a moment to initialize
        time.sleep(5)

        # 2. Start the Flower clients in a loop
        print(f"\n🚀 Starting {NUM_CLIENTS} Flower Clients...")
        for client_id in range(NUM_CLIENTS):
            print(f"   - Starting Client {client_id}")
            client_command = [
                python_executable,
                CLIENT_SCRIPT,
                "--client-id",
                str(client_id),
            ]
            client_process = subprocess.Popen(
                client_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append(client_process)
            print(f"   ✅ Client {client_id} started with PID: {client_process.pid}")
            time.sleep(1) # Stagger client starts slightly

        print("\n🎉 Federated Learning simulation is running...")
        print("Press Ctrl+C to terminate.")

        # Keep the script running and monitor the server process
        # You can also monitor client outputs here if desired
        stdout, stderr = server_process.communicate()
        
        print("\n--- Server Output ---")
        print(stdout)
        if stderr:
            print("\n--- Server Errors ---")
            print(stderr)

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received. Terminating processes...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    finally:
        # 4. Cleanly terminate all child processes
        print("\n🧹 Cleaning up all processes...")
        for process in processes:
            if process.poll() is None:  # Check if the process is still running
                process.terminate()
                process.wait() # Wait for the process to terminate
                print(f"   - Terminated process with PID: {process.pid}")
        print("✅ Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()
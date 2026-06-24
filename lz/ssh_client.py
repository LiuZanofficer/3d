import paramiko
import sys
import argparse

# Configuration
HOSTNAME = 'connect.nma1.seetacloud.com'
PORT = 19699
USERNAME = 'root'
PASSWORD = 'pJEtQajdHPPB'

def run_remote_command(command):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect
        client.connect(HOSTNAME, port=PORT, username=USERNAME, password=PASSWORD, timeout=10)

        # Execute
        print(f"[Remote] Executing: {command}")
        stdin, stdout, stderr = client.exec_command(command)

        # Print Output
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()

        if out:
            print(out)
        if err:
            print(f"STDERR:\n{err}", file=sys.stderr)

    except Exception as e:
        print(f"SSH Connection Error: {e}", file=sys.stderr)
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run command on AutoDL server")
    parser.add_argument("command", help="The command to run on the server")
    args = parser.parse_args()

    run_remote_command(args.command)

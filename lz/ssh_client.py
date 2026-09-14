import os
import paramiko
import sys
import argparse

# Configuration — all read from environment variables; never hardcode credentials.
#   export AUTODL_HOST=connect.xxx.seetacloud.com
#   export AUTODL_PORT=19699
#   export AUTODL_USER=root
#   export AUTODL_PASSWORD=<your-password>
HOSTNAME = os.environ['AUTODL_HOST']
PORT = int(os.environ.get('AUTODL_PORT', '22'))
USERNAME = os.environ['AUTODL_USER']
PASSWORD = os.environ['AUTODL_PASSWORD']

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

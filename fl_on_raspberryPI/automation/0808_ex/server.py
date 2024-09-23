import subprocess

def run_command(command):
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        # Print the output of the command
        print(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        # If the command fails, print the error
        print(f"Error occurred:\n{e.stderr}")

# Commands to run
commands = [
    ['python', 'server.py', '--rounds=100', '--sample_fraction=1.0', '--min_num_clients=35'], # watch only C=12
    ['python', 'server.py', '--rounds=100', '--sample_fraction=1.0', '--min_num_clients=22'], # phone only C=12
    ['python', 'server.py', '--rounds=100', '--sample_fraction=1.0', '--min_num_clients=40'], # re-conduct baseline watch 40
    ['python', 'server.py', '--rounds=100', '--sample_fraction=0.5', '--min_num_clients=40'], # re-conduct baseline watch 20
    ['python', 'server.py', '--rounds=100', '--sample_fraction=1.0', '--min_num_clients=40'], # re-conduct baseline phone 40
    ['python', 'server.py', '--rounds=100', '--sample_fraction=0.5', '--min_num_clients=40'], # re-conduct baseline phone 20
]

for command in commands:
    run_command(command)
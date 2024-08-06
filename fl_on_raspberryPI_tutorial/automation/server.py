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
    ['python', 'server.py', '--rounds=2', '--sample_fraction=0.5', '--min_num_clients=40'],
    ['python', 'server.py', '--rounds=2', '--sample_fraction=0.75', '--min_num_clients=40']
]

for command in commands:
    run_command(command)
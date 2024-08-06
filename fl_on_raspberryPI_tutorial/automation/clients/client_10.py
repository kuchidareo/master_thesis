import subprocess

def run_command(command):
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        # Print the output of the command
        print(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        # If the command fails, print the error
        print(f"Error occurred:\n{e.stderr}")

# Commands to run
commands = [
    'bash experiment_script/wisdm_phone/baseline_10.sh',
    'bash experiment_script/wisdm_phone/baseline_10.sh'
]

for command in commands:
    run_command(command)
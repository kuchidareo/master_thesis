import json
import argparse
import os

def concatenate_json_files(file_list, output_file):
    result = []

    for file_name in file_list:
        with open(file_name, 'r') as file:
            data = json.load(file)
            if isinstance(data, list):
                result.extend(data)
            else:
                result.append(data)

    with open(f"./ex_data/{output_file}", 'w') as output_file:
        json.dump(result, output_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Concatenate multiple JSON files into one JSON file.')
    parser.add_argument('--files', nargs='+', required=True, help='List of JSON files to concatenate.')
    parser.add_argument('--output', required=True, help='Output file name for the merged JSON.')
    
    args = parser.parse_args()
    
    file_list = args.files
    output_file = args.output
    
    concatenate_json_files(file_list, output_file)
    
if __name__ == '__main__':
    main()
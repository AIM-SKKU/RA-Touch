import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate average scores from a JSON file")
    parser.add_argument("--json", type=str, help="Path to the JSON file", required=True)
    return parser.parse_args()

def calculate_dataset_scores(json_file_path):
    try:
        if not os.path.exists(json_file_path):
            print(f"File not found: {json_file_path}")
            return None
        
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        hct_scores = []
        ssvtp_scores = []

        for item in data:
            evaluation = item.get('evaluation')
            image_fp = item.get('image_fp')
            
            if evaluation and image_fp:
                try:
                    # get score from the first element of the string
                    score = float(evaluation.split()[0])                
                except (ValueError, IndexError):
                    print(f"Invalid score format in item: {item}")
                    score = 0
                
                if "hct" in image_fp:
                    hct_scores.append(score)
                elif "ssvtp" in image_fp:
                    ssvtp_scores.append(score)

        def calculate_average(scores):
            return sum(scores) / len(scores) if scores else None

        hct_avg = calculate_average(hct_scores)
        ssvtp_avg = calculate_average(ssvtp_scores)

        total_scores = hct_scores + ssvtp_scores
        total_avg = calculate_average(total_scores)

        print(f"Average Score for 'ssvtp' Dataset: {ssvtp_avg}")
        print(f"Average Score for 'hct' Dataset: {hct_avg}")
        print(f"Total Average Score: {total_avg}")

        return hct_avg, ssvtp_avg, total_avg

    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file: {json_file_path}")
        return None

if __name__ == "__main__":
    args = parse_args()
    json_file_path = args.json
    calculate_dataset_scores(json_file_path)

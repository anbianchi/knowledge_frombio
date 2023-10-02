# Description: Main file for running the pipeline.
import argparse
import time
from modules.report_generation import process_reports_from_dataset, process_reports_from_folder
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process medical reports in two modes.')
    parser.add_argument('--manual', action='store_true', help='Use manually inserted reports in the diagnostic_reports folder.')
    parser.add_argument('--dataset', type=str, help='Specify the path to the dataset file to process reports from.')

    args = parser.parse_args()
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started processing at: {start_datetime}")

    if args.manual:
        process_reports_from_folder()
    elif args.dataset:
        process_reports_from_dataset(args.dataset)
    else:
        print("Please specify a mode: --manual for manually inserted reports or --dataset <path_to_dataset> for processing from a dataset.")

    end_time = time.time()  # End timing here
    total_time = end_time - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"The total time taken for processing is: {total_time} seconds")
    print(f"Ended processing at: {end_datetime}")

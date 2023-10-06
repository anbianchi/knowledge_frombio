import os
import time
import pandas as pd
from modules.preprocessing import clear_reports_folder, preprocess_medical_knowledge, process_reports

def generate_reports(input_file, num_patients=None):
    # Create the 'diagnostic_reports' folder if it doesn't exist
    if not os.path.exists('diagnostic_reports'):
        os.makedirs('diagnostic_reports')

    # Read the CSV file with no truncation
    df = pd.read_csv(input_file)

    # Limit the number of patients if num_patients is provided
    if num_patients:
        df = df.head(num_patients)

    # Generate .txt reports for each patient
    for idx, row in df.iterrows():
        # Use the new headers based on your dataset
        patient_id = row['subject_id']
        print("Patient:",patient_id)
        
        # For each illness_history column, create a separate file if the value is not NaN
        for col in df.columns:
            if "illness_history_" in col and pd.notna(row[col]):
                with open(f'diagnostic_reports/#{patient_id}_{col}.txt', 'w', encoding='utf-8') as f:
                    f.write(str(row[col]))
        #print the number of files in the diagnostic_reports folder
        print("Number of files in the diagnostic_reports folder: ", len(os.listdir('diagnostic_reports')))
        
        preprocessed_reports, filename = preprocess_medical_knowledge()
        process_reports(preprocessed_reports, filename=filename)

        # Clear the reports folder for the next patient
        clear_reports_folder()
        
        time.sleep(3)
        

    print(f"Reports processing completed.")
    
def process_reports_from_dataset(dataset_file):
    """
    Process reports from the specified dataset file.
    """
    generate_reports(input_file=dataset_file, num_patients=None)
    #preprocessed_reports, filename = preprocess_medical_knowledge()
    #process_reports(preprocessed_reports, filename)
    
def process_reports_from_folder():
    """
    Process reports that are manually inserted into the 'diagnostic_reports' folder.
    """
    preprocessed_reports, filename = preprocess_medical_knowledge()
    process_reports(preprocessed_reports, filename=filename)
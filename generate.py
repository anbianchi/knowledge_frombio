import pandas as pd
import re

input_file = "discharge.csv"
output_file = "processed_patients.csv"

# Function to extract 'History of Present Illness' from the text, bounded by "Past Medical History"
def extract_history_of_present_illness(text):
    if pd.isna(text):
        return None
    if re.search(r'History of Present Illness:', text) and re.search(r'Past Medical History', text):
       match = re.search(r'History of Present Illness:\s*(.*?)(?=\nPast Medical History)', text, re.DOTALL)
       return match.group(1).strip() if match else None
    else:    
       return None

# Read the CSV file
df = pd.read_csv(input_file)

# Create a 'count' column that will keep track of the hospitalization number for each patient
df['count'] = df.groupby('subject_id').cumcount() + 1

# Pivot the dataframe to get the desired structure
pivot_df = df.pivot(index='subject_id', columns='count', values='text')
pivot_df = pivot_df.applymap(extract_history_of_present_illness)

# Filter out rows with fewer than two non-null entries in the "illness_history_#number" columns
pivot_df = pivot_df[pivot_df.count(axis=1) >= 2]

# Rename the columns for clarity
pivot_df.columns = [f"illness_history_{i}" for i in pivot_df.columns]

# Save the result to the output file
pivot_df.reset_index().to_csv(output_file, index=False)

# Print the number of processed patients
print(f"The creation of the dataset is over. Processed {len(pivot_df)} patients.")

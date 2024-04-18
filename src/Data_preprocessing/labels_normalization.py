import pandas as pd

disease_categories = {
        'Atelectasis': 0,
        'Cardiomegaly': 1,
        'Effusion': 2,
        'Infiltrate': 3,
        'Mass': 4,
        'Nodule': 5,
        'Pneumonia': 6,
        'Pneumothorax': 7,
        'Consolidation': 8,
        'Edema': 9,
        'Emphysema': 10,
        'Fibrosis': 11,
        'Pleural_Thickening': 12,
        'Hernia': 13,
        'no_finding': 14,
        }
# Load the CSV file
file_path = 'Data_csv\data_Data_Entry_2017_v2020.csv'
data = pd.read_csv(file_path)

# Create a new column 'labels' to store the first label before '|'
data['labels'] = data['Finding Labels'].apply(lambda x: x.split('|')[0].strip())

# Split the "Finding Labels" column and explode it into separate rows
exploded_data = data.assign(FindingLabels=data['Finding Labels'].str.split('|')).explode('FindingLabels')

# Move the 'labels' column to the end
selected_columns = data[['Image Index', 'Finding Labels', 'Patient ID','labels']]

# Save the exploded data to a new CSV file or use it for further analysis
output_path = 'Data_csv/exploded_data_with_labels.csv'
selected_columns.to_csv(output_path, index=False)

print("Data has been processed and saved to:", output_path)

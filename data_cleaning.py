import pandas as pd

# Assuming your dataset is in a CSV file named 'updated_dataset.csv'
# Adjust the file name and path accordingly if it's in a different format or location
dataset = pd.read_csv('updated_dataset.csv')

#dataset.dropna(subset=['URL', 'Label'], inplace=True)

# Sort the dataset based on the 'urls' column
dataset.sort_values(by='URL', inplace=True)

# Clean the dataset by removing duplicate URLs while preserving their corresponding labels
cleaned_dataset = dataset.drop_duplicates(subset='URL', keep='first')


cleaned_dataset.to_csv('cleaned_data_new.csv', index=False)

# Print the cleaned dataset
print("Cleaned dataset saved to 'cleaned_data_new.csv'")

#import pandas as pd

# Read the dataset into a DataFrame
url_data = pd.read_csv('cleaned_data.csv')





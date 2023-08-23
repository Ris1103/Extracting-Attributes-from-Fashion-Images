import pandas as pd

# Load the submission.csv file
submission_df = pd.read_csv('submission.csv')

# Remove the "data\" part from the file_name column
submission_df['file_name'] = submission_df['file_name'].apply(lambda x: x.split("\\")[-1])

# Save the modified DataFrame back to submission.csv
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' updated successfully.")
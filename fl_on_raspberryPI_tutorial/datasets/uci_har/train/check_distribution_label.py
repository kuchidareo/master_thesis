import pandas as pd

# Read the y_train.txt file
y_train = pd.read_csv('./y_train.txt', header=None, names=['Activity'])

# Read the subject_train.txt file
subject_train = pd.read_csv('./subject_train.txt', header=None, names=['Subject'])

# Concatenate the DataFrames along the columns
concatenated_df = pd.concat([subject_train, y_train], axis=1)

# Group by 'Subject' and get the distribution of 'Activity' for each subject
activity_distribution = concatenated_df.groupby('Subject')['Activity'].value_counts().unstack().fillna(0)

# Print the distribution
print(activity_distribution)
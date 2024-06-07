'''
This file contains all of the data cleaning and preparation needed before running a model.

In the 7 Step Chollet Process, this file covers steps 1 - 3.

STEP 1: GET SOME DATA

The data is fishing data coming from: https://globalfishingwatch.org/data-download/datasets/public-training-data-v1

Observational in nature, there are many reported features that were collected before determining whether the ship
was fishing or not fishing.


STEP 2: DETERMINE A MEASURE OF SUCCESS

For this project, the measure of success will be final model accuracy.


STEP 3: PREPARE SOME DATA

As follows, we load it in, and then use a variety of techniques to clean it. All in all, we reduce +14 million observations
down to just over 300 thousand. The steps to do so are covered in the rest of this script.
'''

# Install necessary packages
import pandas as pd 

# # Load in all of the data
# df1 = pd.read_csv('/remote_home/EENG645A_FinalProject-1/raw_data/fixed_gear.csv')
# df2 = pd.read_csv('/remote_home/EENG645A_FinalProject-1/raw_data/pole_and_line.csv')
# df3 = pd.read_csv('/remote_home/EENG645A_FinalProject-1/raw_data/purse_seines.csv')
# df4 = pd.read_csv('/remote_home/EENG645A_FinalProject-1/raw_data/trawlers.csv')
# df5 = pd.read_csv('/remote_home/EENG645A_FinalProject-1/raw_data/trollers.csv')
# df6 = pd.read_csv('/remote_home/EENG645A_FinalProject-1/raw_data/unknown.csv')
# print(f"All data loaded in.")

# # Data preprocessing, all done to get the combined.csv file

# # Add a new column in each file with the boat type
# df1['boat_type'] = 'fixed_gear'
# df2['boat_type'] = 'pole_and_line'
# df3['boat_type'] = 'purse_seines'
# df4['boat_type'] = 'trawlers'
# df5['boat_type'] = 'trollers'
# df6['boat_type'] = 'unknown'
# print(f"All boat_type columns populated.")

# # Append all the files together
# df_combined = pd.concat([df1, df2], ignore_index=True)
# df_combined = pd.concat([df_combined, df3], ignore_index=True)
# df_combined = pd.concat([df_combined, df4], ignore_index=True)
# df_combined = pd.concat([df_combined, df5], ignore_index=True)
# df_combined = pd.concat([df_combined, df6], ignore_index=True)
# print(f"All dataframes combined as one.")

# # # Print the combined dataframe to make sure it loaded correctly
# # print(df_combined)

# Save the combined DataFrame back to a CSV file
# df_combined.to_csv('/remote_home/EENG645A_FinalProject-1/processed_data/combined.csv', index=False)
# print(f"All data combined into one data frame.")

df = pd.read_csv('/remote_home/EENG645A_FinalProject-1/processed_data/combined.csv')
print(df)

# Count occurrences where is_fishing is greater than 0 (Places where fishing is occuring)
greater_than_zero_count = (df['is_fishing'] > 0).sum()
# Count occurrences where is_fishing is 0 (Places where fishing is not occuring)
zero_count = (df['is_fishing'] == 0).sum()
# Count occurrences where is_fishing is -1 (Places with no data)
less_than_zero_count = (df['is_fishing'] < 0).sum()

print("Count of is_fishing > 0:", greater_than_zero_count)
print("Count of is_fishing == 0:", zero_count)
print("Count of is_fishing < 0:", less_than_zero_count)

''' 
Output from prints:
Count of is_fishing > 0: 117709
Count of is_fishing == 0: 216405
Count of is_fishing < 0: 14278557

This is indicative of major class imbalance. Thankfully, we do not need the data that represents no data collected, so
by removing it we can have balanced data that also informs us of where fishing did and did not take place.
'''

# # Only keep rows where is_fishing is greater than 0 (remove the rows with no data) 
# df_filtered = df[df['is_fishing'] >= 0]
# df_filtered.to_csv('/remote_home/EENG645A_FinalProject-1/processed_data/filtered.csv', index=False)

# # Count occurrences where is_fishing is greater than 0 (Places where fishing is occuring)
# greater_than_zero_count = (df_filtered['is_fishing'] > 0).sum()
# # Count occurrences where is_fishing is 0 (Places where fishing is not occuring)
# zero_count = (df_filtered['is_fishing'] == 0).sum()
# # Count occurrences where is_fishing is -1 (Places with no data)
# less_than_zero_count = (df_filtered['is_fishing'] < 0).sum()

# print("Count of is_fishing > 0:", greater_than_zero_count)
# print("Count of is_fishing == 0:", zero_count)
# print("Count of is_fishing < 0:", less_than_zero_count)

''' 
Output from prints:
Count of is_fishing > 0: 117709
Count of is_fishing == 0: 216405
Count of is_fishing < 0: 0

This is what we expect, so we can now continue!
'''

# df = pd.read_csv('/remote_home/EENG645A_FinalProject-1/processed_data/filtered.csv')

# # Count occurrences where is_fishing is 1 (Places where fishing always reported as occuring)
# greater_than_zero_count = (df['is_fishing'] == 1).sum()
# # Count occurrences where is_fishing is 0 (Places where fishing is not occuring)
# zero_count = (df['is_fishing'] == 0).sum()
# print("Count of is_fishing == 1:", greater_than_zero_count)
# print("Count of is_fishing == 0:", zero_count)

''' 
Output from prints:
Count of is_fishing == 1: 109335
Count of is_fishing == 0: 216405

Remember from above that the number of values that were above 0 was 117709, but here we see the number of rows where values
equal exactly 1 are 109335. This means that there are 8374 rows that have a value between 0 and 1. The dataset says that
a value between 0 and 1 represents the proportion of reports for a vessel that was seen fishing. For simpicity sake,
we will round these values up to 1 because they were all reported fishing at least once. This will reduce our final model
to a binary classification of was a vessel fishing (yes or no).
'''

# df.loc[(df['is_fishing'] > 0) & (df['is_fishing'] < 1), 'is_fishing'] = 1
# # Count occurrences where is_fishing is 1 (Places where fishing always reported as occuring)
# greater_than_zero_count = (df['is_fishing'] == 1).sum()
# # Count occurrences where is_fishing is 0 (Places where fishing is not occuring)
# zero_count = (df['is_fishing'] == 0).sum()
# print("Count of is_fishing == 1:", greater_than_zero_count)
# print("Count of is_fishing == 0:", zero_count)

''' 
Output from prints:
Count of is_fishing == 1: 117709
Count of is_fishing == 0: 216405

Now all the data either corresponds to an instance of a vessel fishing or not fishing. Now we need to check for missing
values or values that do not make sense in the data (extremes or unrealistic attributes).
'''

# # Save to a new csv file
# df.to_csv('/remote_home/EENG645A_FinalProject-1/processed_data/bin_response.csv', index=False)

# df = pd.read_csv('/remote_home/EENG645A_FinalProject-1/processed_data/bin_response.csv')

# # For every column
# for column in df.columns:   
#         # Print the column name 
#         print(f"Column: {column}")

#         # Check for missing values
#         missing_values = df[column].isna().sum()
#         print(f"Missing values: {missing_values}")
        
#         # Check for zeros (Not applicable to distance_from_shore, distance_from_port, speed, course, or is_fishing)
#         zeros = (df[column] == 0).sum()
#         print(f"Zero values: {zeros}")
        
#         # Check for extreme values (Example: Consider values below 1st percentile or above 99th percentile as extreme)
#         if pd.api.types.is_numeric_dtype(df[column]):
#             lower_bound = df[column].quantile(0.01)
#             upper_bound = df[column].quantile(0.99)
#             extreme_values = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
#             print(f"Extreme values: {extreme_values}")
        
#         print("")

'''
Column: mmsi
Missing values: 0
Zero values: 0
Extreme values: 1085

Column: timestamp
Missing values: 0
Zero values: 0
Extreme values: 6684

Column: distance_from_shore
Missing values: 0
Zero values: 142155
Extreme values: 3341

Column: distance_from_port
Missing values: 0
Zero values: 7815
Extreme values: 3340

Column: speed
Missing values: 2
Zero values: 133624
Extreme values: 3279

Column: course
Missing values: 2
Zero values: 25869
Extreme values: 207

Column: lat
Missing values: 0
Zero values: 0
Extreme values: 6684

Column: lon
Missing values: 0
Zero values: 0
Extreme values: 6544

Column: is_fishing
Missing values: 0
Zero values: 216405
Extreme values: 0

Column: source
Missing values: 0
Zero values: 0

Column: boat_type
Missing values: 0
Zero values: 0


The columns that have no discernable isssue are is_fishing (0's mean no fishing was observed), source, and boat_type. To
fix the rows with missing values, because there were only two they were simply removed. All other columns that had a 0
in any of their rows were determined to be logical and thus correctly inputted as a 0. For example, the speed column has
133624 rows with a 0 in them. This makes sense because, for example, a boat can have a speed of 0 (moored at a dock or 
anchored at sea), and a course can be 0 (heading of true North). So now we need to take a closer look at the extreme
values that were observed. 
'''
# for column in df.columns:
#     min_value = df[column].min()
#     max_value = df[column].max()
#     print(f"Column: {column}")
#     print(f"  Smallest value: {min_value}")
#     print(f"  Largest value: {max_value}")
#     print("\n")

# # Count instances of 'course' feature that are greater than 360
# count_greater_than_360 = (df['course'] > 360).sum()

# # Print the count
# print("Count of 'course' values greater than 360:", count_greater_than_360)

# extreme_indices = set()
    
# if pd.api.types.is_numeric_dtype(df['mmsi']):
#     lower_bound = df['mmsi'].quantile(0.01)
#     upper_bound = df['mmsi'].quantile(0.99)
#     extreme_condition = (df['mmsi'] < lower_bound) | (df['mmsi'] > upper_bound)
#     extreme_indices.update(df[extreme_condition].index)
    
# extreme_value_indices = list(extreme_indices)

# # Print the indices of rows with extreme values
# print("Indices of rows with extreme values:", extreme_value_indices)

''' 
This print output is not included. It gave me an idea of which rows to look at to find extreme values, so I could manually
inspect the dataset. Taking the column's one at a time, MMSI is a ship's identifier. As long as there are no missing values,
extremes do not provide valuable information for identifying numbers.

For timestamp, there were a variety of UNIX timestamps covering multiple years worth of data collection. UNIX is represented
in seconds, so to cover years, you would have some numbers be a hundred times greater than another number. All numbers were
between 1.3 trillion and 1.5 trillion, so no issues were determined to exist.

For distance_from_shore, some ships were out in deep ocean waters while many were moored at a dock. This variance in scale
provided a way for extremes to come about. When looking deeper into the issue, the largest distance from shore was
2854625 meters (2854.6 km). In reality, the point that is farthest from shore in any direction is Point Nemo in the Pacific
Ocean, which is at least 2688 kilometers from land in any direction. This means that the value was either recorded
incorrectly, or whoever reported it did not measure to the closest point of land. Perhaps they recorded to their own 
country's shoreline. I decided to keep these values as there is still some merit into it, but perhaps the final model could be
more accurate if complete case analysis was done on any row with a value above 2688000 in its column.

The feature distance_from_port had the same issue as distance_from_shore, where some boats were logged and reported to be in 
a port so their distance was 0, while some were out in open ocean. However, even worse than the distance_from_shore feature,
the greatest value here was 3836963.25 meters (3836.9 km)! That's an extra 1000 kilometers from Point Nemo to the next 
closest port, which is very unrealistic. This suggests that it was not an issue with being recorded incorrectly, but an
issue with measurement. However, because the website offers no standardized way to measure distances to ports,

For speed, many boats were anchored or moored so their speed was 0. This brought the average down, so if a boat was seen
speeding across open water, it looked like an extreme value. Nothing was so fast though that it appeared to be incorrectly
recorded.

For course, the largest value observed was 511. From my understanding, course should only be able to take on values between
0 and 360, which means that any measurement that is 360 or greater is too large and is an incorrect measurement. Taking 
count, the number of observations that were greater or equal to 360 were 207 which is coincidentally (or maybe not) also 
the number of extreme values identified. Complete case analysis was done to remove all these observations from the final
dataset.

For latitude, it turned out that the vast majority of values were around the equator (-15 degrees to
15 degrees). A possible reason for this is that the warmer waters not only promote more fishing vessels to be out on the water,
but that there are also more ships out on the water who are willing to create log and submit the reports of other ships and
their activity, as opposed to colder waters. 

While latitude can take on values from -90 to 90, longitude takes on values from -180 to 180. There were longitude values from 
a variety of places around the world, but the values that kept popping up as extremes were about 170. This is a valid line of 
longitude, so I looked at a map to see if I could gain any insight. It turns out this is approximately the middle of the 
Pacific Ocean, so boats going out this far are likely sparse and distanced from each other. Additionally, anyone who was 
helping to collect data for this report likely would not have wanted to venture here to check for fishing activity. Thus,
all extremes for latitude and longitude are accounted for. While the data may have observation bias, it is otherwise clean.

Finally for source, some of the sources of data collection were marked as 'false positives', the meaning of which is unclear.
There were 8000 of these values, so all of them were removed as well, just to ensure that our data had nothing icky
going on behind the scenes.

And then, the data was clean!
'''

# df = pd.read_csv('/remote_home/EENG645A_FinalProject-1/processed_data/bin_response.csv')

# # Remove rows where 'course' is 360 or greater
# df_final = df[df['course'] < 360]

# df_final2 = df_final[df_final['source'] != 'false_positives']

# # Optionally, save the updated DataFrame back to the CSV file
# df_final2.to_csv('/remote_home/EENG645A_FinalProject-1/processed_data/clean.csv', index=False)

''' We can now start training! '''



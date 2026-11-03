# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================

PH_LIMITS = (6.8, 7.2)
TEMPERATURE_LIMITS = (35.0, 37.0)

#to ensure that the regression included the most representative training data, the batch IDs were
#run through and randomly selected. The individual initial conditions were then compared to the
#previous batches to ensure that there was no repeatability, although, it's possible that this
#could have also been helpful to ensure that the results were consistent throughout runs.

import os
import pandas as pd
import random

df = pd.read_csv("datasets/initial_conditions.csv")


#===========================================
#   Code Initially Used to Find Batch IDs
#===========================================

# # Store selected batches and conditions
# selected_batches = []
# selected_conditions = set()
#
# # Continue selecting until we have 30 unique conditions
# while len(selected_batches) < 30:
#
#     # Randomly select a row
#     row = df.sample(n=1).iloc[0]
#
#     # Identify the experimental conditions
#     conditions = (
#         row["Glucose [mM]"],
#         row["Glutamine [mM]"],
#         row["VCD [cells/mL]"]
#     )
#
#     # Check if these conditions have already been selected
#     if conditions not in selected_conditions:
#
#         selected_conditions.add(conditions)
#         selected_batches.append(row["Batch ID"])
#
# selected_batch_ids=sorted(selected_batches)

selected_batch_ids=[36 ,108, 133, 142, 183, 187, 206, 215, 260, 349, 373, 383, 396, 430, 530, 547, 571, 591, 615, 648, 657, 661, 693, 725, 771, 810, 813, 832, 925, 1020]

#finding initial conditions of selected datasets

# List of column values to filter by
target_conditions = selected_batch_ids

# Filter rows where the 'Age' column matches any value in the list
result = df[df['Batch ID'].isin(target_conditions)]

result.to_csv(
    os.path.join("tables", "table_01_initial_conditions.csv"),
    index=False
)
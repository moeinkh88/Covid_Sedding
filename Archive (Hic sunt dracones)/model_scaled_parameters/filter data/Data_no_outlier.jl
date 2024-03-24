# Your dataset
data = TrueD

# Calculate Q1, Q3, and IQR
Q1 = quantile(data, 0.25)
Q3 = quantile(data, 0.75)
IQR = Q3 - Q1

# Calculate the bounds
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

# Filter the dataset to remove outliers
filtered_data = filter(x -> x >= lower_bound && x <= upper_bound, data)

# Display the filtered dataset
println(filtered_data)
scatter(TrueD)
scatter!(filtered_data)

##test shitty code

using CSV
using DataFrames
using Statistics

# Load the dataset
dataset_R = CSV.read("time_series_covid19_recovered_global.csv", DataFrame) # all data of Recover
RData=dataset_R[dataset_R[!,2].=="South Africa",70:250]

# Convert to Float64 and calculate the differences
TrueR0=diff(Float64.(Vector(RData[1,:])))

# Calculate Q1, Q3, and IQR
Q1 = quantile(TrueR0, 0.25)
Q3 = quantile(TrueR0, 0.75)
IQR = Q3 - Q1

# Calculate the bounds
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

# Initialize a copy of TrueR0 to modify
TrueR1 = copy(TrueR0)

# Identify indices of outliers
outlier_indices = findall(x -> x < lower_bound || x > upper_bound, TrueR0)

# Replace outliers with the mean of their neighbors
for idx in outlier_indices
    if idx == 1 # Handle first element
        TrueR1[idx] = TrueR1[idx + 1]
    elseif idx == length(TrueR0) # Handle last element
        TrueR1[idx] = TrueR1[idx - 1]
    else
        TrueR1[idx] = mean([TrueR1[idx - 1], TrueR1[idx + 1]])
    end
end

# TrueR1 now contains the original data with outliers replaced by the mean of their neighbors

scatter(TrueR0)
scatter!(TrueR1)


##

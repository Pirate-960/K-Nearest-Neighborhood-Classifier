import pandas as pd
import random

# Function to generate a large synthetic "Play Tennis" dataset
def generate_large_dataset(file_path, num_records=1000):
    # Possible values for each feature
    outlook_options = ["Sunny", "Overcast", "Rain"]
    temperature_options = ["Hot", "Mild", "Cool"]
    humidity_options = ["High", "Normal"]
    wind_options = ["Weak", "Strong"]
    play_tennis_options = ["Yes", "No"]

    # Generate random records
    data = []
    for _ in range(num_records):
        record = {
            "Outlook": random.choice(outlook_options),
            "Temperature": random.choice(temperature_options),
            "Humidity": random.choice(humidity_options),
            "Wind": random.choice(wind_options),
            "PlayTennis": random.choice(play_tennis_options),
        }
        data.append(record)

    # Create a DataFrame and save to JSON
    df = pd.DataFrame(data)
    df.to_json(file_path, orient='records', indent=4)
    print(f"Large dataset with {num_records} records saved to {file_path}")

# Generate and save the dataset
generate_large_dataset("Dataset/large_play_tennis_dataset.json", num_records=1000)
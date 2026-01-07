import pandas as pd

def write_to_csv(data, file_path):
    # Save the actual and predicted captions to CSV using Pandas
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
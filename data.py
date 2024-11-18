import pandas as pd
import random

# Function to generate synthetic company names
def generate_company_names(num_entries):
    prefixes = ["Tech", "Data", "Global", "Info", "Alpha", "Smart", "Mega", "Inno"]
    suffixes = ["Corp", "Inc", "Solutions", "Systems", "Labs", "Ventures", "Holdings"]
    return [f"{random.choice(prefixes)} {random.choice(suffixes)}" for _ in range(num_entries)]

# Function to create synthetic data
def create_synthetic_data(num_entries=50):
    # Generate synthetic entities (e.g., company names)
    companies = generate_company_names(num_entries)
    
    # Create a DataFrame with the synthetic data
    df = pd.DataFrame({
        "Company": companies,
    })
    
    # Save to CSV
    df.to_csv("synthetic_data.csv", index=False)
    print("Synthetic data saved to 'synthetic_data.csv'")
    return df

# Generate and preview the synthetic data
synthetic_df = create_synthetic_data(50)
print(synthetic_df.head())

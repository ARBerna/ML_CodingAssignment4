import pandas as pd

df = pd.read_csv("dog_breeds.csv")

def convert_range(value):
    if isinstance(value, str) and "-" in value:
        low, high = value.split("-")
        return (float(low) + float(high)) / 2
    else:
        try:
            return float(value)
        except:
            return None

df["Height_fixed"] = df["Height (in)"].apply(convert_range)
df["Longevity_fixed"] = df["Longevity (yrs)"].apply(convert_range)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df["Longevity_normalized"] = scaler.fit_transform(df[["Longevity_fixed"]])

print(df[[
    "Height (in)", "Height_fixed",
    "Longevity (yrs)", "Longevity_fixed",
    "Longevity_normalized"
]].head(30))
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("train.csv")
print(df.head())

catigorical_columns = ['Sex', 'Embarked', 'Pclass']

encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')

encoded_array = encoder.fit_transform(df[catigorical_columns])
encoded_columns = encoder.get_feature_names_out(catigorical_columns)

encoded_df = pd.DataFrame(encoded_array, columns = encoded_columns)

final_df = pd.concat([df.drop(columns = catigorical_columns), encoded_df], axis = 1)

print(final_df.head())
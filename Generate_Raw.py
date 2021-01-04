import os
import pandas as pd
with open('data_raw/raw.csv', 'w+') as f:
    for root, dirs, files in os.walk("data_origin"):
        for file in files:
            if "txtoriginal" in file:
                filepath = os.path.join(root, file)
                content = open(filepath, 'r', encoding='utf-8').read().strip()
                df = pd.DataFrame([content])
                df.to_csv("data_raw/raw.csv", mode='a',
                          header=False, index=False)

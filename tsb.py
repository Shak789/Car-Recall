import requests
import time
import os
import pandas as pd

if os.path.exists('TSBSMC.txt'):
    os.remove('TSBSMC.txt')

url = "https://static.nhtsa.gov/odi/ffdd/tsbs/TSBS_RECEIVED_2025-2026.zip"

response = requests.get(
    url,
    timeout=300,
    headers={'User-Agent': 'Mozilla/5.0'}
)

with open('TSBSMC.txt', 'wb') as f:
    f.write(response.content)

print(f"Done. Size: {os.path.getsize('TSBSMC.txt') / 1024 / 1024:.1f} MB")

# Try reading it
import pandas as pd
try:
    tsb_df = pd.read_csv(
        'TSBSMC.txt',
        sep='\t',
        encoding='latin-1',
        on_bad_lines='skip',
        engine='python'
    )
    print(tsb_df.shape)
    print(tsb_df.head())
except Exception as e:
    print(f"Error: {e}")
    # Try reading first few raw lines to see format
    with open('TSBSMC.txt', 'r', encoding='latin-1', errors='replace') as f:
        for i, line in enumerate(f):
            print(repr(line))
            if i > 5:
                break
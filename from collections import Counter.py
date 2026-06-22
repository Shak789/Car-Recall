from collections import Counter
import pandas as pd
import re

df = pd.read_csv('words.csv')

def parse_phrases(text):
    if not isinstance(text, str):
        return []
    return [p.strip() for p in text.split(",")]

df["top_words"] = df["top_words"].apply(parse_phrases)


NOISE_PATTERNS = [
    # Numbers and dates
    r'\b\d{4,5}\b',                    # zip codes and years
    r'\b\d+,\d+\s*miles?\b',          # mileage like "32,000 miles"
    r'\b\d+k?\s*miles?\b',            # mileage like "32k miles"
    
    # Months and dates
    r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
    r'\b(january|february|march|april|june|july|august|september|october|november|december)\b',
    
    # Brand names — expand this significantly
    r'\b(acura|ford|toyota|honda|bmw|jeep|kia|chevrolet|chevy|gmc|dodge|ram|'
    r'chrysler|jeep|nissan|hyundai|kia|subaru|mazda|volvo|audi|mercedes|benz|'
    r'volkswagen|vw|porsche|lexus|infiniti|cadillac|buick|lincoln|tesla|rivian|'
    r'mitsubishi|suzuki|isuzu|pontiac|saturn|oldsmobile)\b',
    
    # Generic vehicle nouns
    r'\b(vehicle|car|automobile|cars|vehicles|truck|suv|sedan|coupe|van|minivan)\b',
    
    # State codes and locations
    r'\b(va|nj|ca|tx|fl|ny|pa|oh|il|ga|nc|mi|wa|az|ma|co|tn|in|mo|md|wi)\b',
    
    # NHTSA boilerplate language
    r'\b(contact|owns|recall|nhtsa|dealer|dealership|notified|manufacturer|'
    r'complaint|reported|report|tl|vin|mileage|failure|occurred|approximately)\b',
    
    # Generic filler words in complaints
    r'\b(said|told|stated|called|went|came|got|felt|heard|noticed|happened|'
    r'started|stopped|tried|asked|brought|took|left|used|made|found)\b',
    
    # Time references
    r'\b(today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    r'\b(morning|afternoon|evening|night|week|month|year|ago|later|since|before|after)\b',
    
    # Administrative/legal language
    r'\b(warranty|service|repair|technician|mechanic|inspection|replaced|replacement|'
    r'diagnosed|diagnosis|estimated|estimate|cost|paid|charge|fix|fixed)\b',
]

def is_noise(phrase):
    phrase = phrase.lower()
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, phrase):
            return True
    return False

def filter_keyphrases(phrases):
    return [p for p in phrases if not is_noise(p)]


df["top_words_filtered"] = df["top_words"].apply(filter_keyphrases)


# step 1: flatten all keyphrases across all vehicles into one list
all_phrases = [
    phrase
    for phrases in df["top_words_filtered"]
    for phrase in phrases
]

# step 2: count most common
phrase_counts = Counter(all_phrases)
top_phrases = [phrase for phrase, count in phrase_counts.most_common(100)]
print(top_phrases)

print(df["top_words"].iloc[0] )

# how many unique vehicles contributed to these phrases
print(f"Total vehicles: {len(df)}")

# check phrase distribution across components
print(df["top_words"].apply(lambda x: 
    any("brake" in p for p in x)).sum(), "vehicles with brake phrases")
print(df["top_words"].apply(lambda x: 
    any("engine" in p for p in x)).sum(), "vehicles with engine phrases")
print(df["top_words"].apply(lambda x: 
    any("airbag" in p for p in x)).sum(), "vehicles with airbag phrases")
print(df["top_words"].apply(lambda x: 
    any("steer" in p for p in x)).sum(), "vehicles with steering phrases")
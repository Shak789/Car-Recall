import pandas as pd
import numpy as np

nhtsa_headers = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH",
    "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE",
    "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES", "CDESCR", "CMPL_TYPE",
    "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE",
    "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE",
    "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE",
    "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY",
    "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN",
    "MEDICAL_ATTN", "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT", "VEHICLE_OPERATOR"
]

complaint_data_10_14 = pd.read_csv('Complaint_2010_2014.csv', header = None, names = nhtsa_headers)
complaint_data_15_19 = pd.read_csv('Complaint_2015_2019.csv', header = None, names = nhtsa_headers)
complaint_data_20_24 = pd.read_csv('Complaint_2020_2024.csv', header = None, names = nhtsa_headers)
complaint_data_25_26 = pd.read_csv('Complaint_2025_2026.csv', header = None, names = nhtsa_headers)

complaint_data_combined = pd.concat([complaint_data_10_14, complaint_data_15_19, complaint_data_20_24, complaint_data_25_26])

'''
recall_data_10_14 = pd.read_csv('Recall_2010_2014.csv')
recall_data_15_19 = pd.read_csv('Recall_2015_2019.csv')
recall_data_20_24 = pd.read_csv('Recall_2020_2024.csv')
recall_data_25_26 = pd.read_csv('Recall_2025_2026.csv')

recall_data_combined = pd.concat([recall_data_10_14, recall_data_15_19, recall_data_20_24, recall_data_25_26])
'''

recall_data_combined = pd.read_csv('recalled_2010_2020.csv')

complaint_data_combined["YEARTXT"] = pd.to_numeric(complaint_data_combined["YEARTXT"], errors="coerce")
complaint_data_combined = complaint_data_combined.dropna(subset=["YEARTXT"])
complaint_data_combined["YEARTXT"] = complaint_data_combined["YEARTXT"].astype(int)
complaint_data_combined = complaint_data_combined.loc[complaint_data_combined['YEARTXT'] != 9999]

'''
check = ['SUSPENSION', 'EXTERIOR LIGHTING']
complaint_data_combined = complaint_data_combined.loc[
    complaint_data_combined['COMPDESC'].str.upper().str.contains(
        '|'.join(check), na=False
    )
]
'''

PASSENGER_MAKES = {
    "ACURA", "AUDI", "BMW",
    "BUICK", "CADILLAC", "CHEVROLET", "CHRYSLER", "DODGE", 
    "FORD", "GMC", "HONDA", "HUMMER", "HYUNDAI",
    "INFINITI", "JAGUAR", "JEEP", "KIA", "LAND ROVER",
    "LEXUS", "LINCOLN", "MAZDA",
    "MERCEDES-BENZ", "MERCEDES", "MINI",
    "MITSUBISHI", "NISSAN",
    "PORSCHE", "RAM",
    "SUBARU", "TESLA", "TOYOTA", "VOLKSWAGEN", "VOLVO"
}

#complaint_data_combined['MILES'] = complaint_data_combined['MILES'].fillna(0)
complaint_data_combined = complaint_data_combined[complaint_data_combined["MAKETXT"].str.upper().isin(PASSENGER_MAKES)]
complaint_data_combined.loc[complaint_data_combined["MAKETXT"] == 'MERCEDES', "MAKETXT"] = "MERCEDES-BENZ"

for col in ["CRASH", "FIRE", "MEDICAL_ATTN", "VEHICLES_TOWED_YN", "POLICE_RPT_YN"]:
    complaint_data_combined[col] = complaint_data_combined[col].map({"Y": 1, "N": 0}).fillna(0).astype(int)


numeric_dates = pd.to_numeric(complaint_data_combined["LDATE"], errors="coerce")
complaint_data_combined["LDATE"] = numeric_dates.astype("Int64").astype(str)

complaint_data_combined['FAILDATE'] = pd.to_datetime(complaint_data_combined['FAILDATE'].astype(str), format='%Y%m%d', errors='coerce')

complaint_data_combined['LDATE'] = pd.to_datetime(complaint_data_combined['LDATE'].astype(str), format='%Y%m%d', errors='coerce')


complaint_data_combined = complaint_data_combined.rename(columns = {
    "MAKETXT": "MAKE",
    "MODELTXT": "MODEL",
    "YEARTXT": "YEAR"})

complaint_data_combined.drop_duplicates(subset = ["ODINO"])

complain_data_combined_scoring = complaint_data_combined

raw_data = complaint_data_combined[(complaint_data_combined["YEAR"] >= 2010) & (complaint_data_combined["YEAR"] <= 2020)]

raw_data.to_csv("complaints_raw.csv")

COMPONENT_LIST = [
'ELECTRICAL SYSTEM',
'ENGINE',
'POWER TRAIN',
'AIR BAGS',
'STEERING']


complaint_data_combined['RELEASE_DATE'] = pd.to_datetime(
    complaint_data_combined['YEAR'].astype(str) + '-01-01'
)

complaint_data_combined['within_12_months'] = (
    (complaint_data_combined['LDATE'] - complaint_data_combined['RELEASE_DATE']).dt.days.between(0, 365)
)



def days_to_peak(date_series):
    if len(date_series) < 2:
        return 0

    daily_counts = date_series.value_counts()

    if daily_counts.empty:
        return 0

    max_count = daily_counts.max()
    peak_days = daily_counts[daily_counts == max_count].index.sort_values()

    if len(peak_days) == 0:
        return 0

    peak_day = peak_days[0]
    first_day = date_series.min()

    # Return the exact integer number of days between them
    return (peak_day - first_day).days


SAFETY_KEYPHRASES = [
    "sunroof exploded",
    "sunroof shattered",
    "sunroof exploded driving",
    "steering wheel seized",
    "steering failed",
    "power steering failed",
    "brake failure",
    "emergency braking",
    "emergency brake",
    "windshield crack",
    "windshield cracked",
    "tire blew",
    "highway tire separated",
    "airbag",
    "air bags",

    # Steering — well represented, all important
    'steering wheel seized',
    'steering failed',
    'power steering failed',
    'lost power steering',
    'steering wheel locked',
    'steering stopped',

    # Brakes — critical
    'brake failure',
    'emergency braking',
    'emergency brake',
    'brake fluid',

    # Glazing/windows — clearly a pattern
    'sunroof exploded',
    'sunroof exploded driving',
    'sunroof shattered',
    'rear window exploded',
    'windshield crack',
    'windshield cracked',
    'cracked windshield',

    # Airbags/restraints
    'airbags deployed',
    'passenger airbag light',
    'seat belts',

    # Tires
    'tire blew',
    'tire failure',

    'steering wheel seized',
    'steering failed',
    'power steering failed',
    'lost power steering',
    'steering wheel locked',

    # Brakes
    'brake failure',
    'emergency braking',
    'emergency brake',
    'emergency brake engaged',
    'brake fluid',
    'auto stop',
    'auto stop start',

    # Glazing/windows
    'sunroof exploded',
    'sunroof exploded driving',
    'sunroof shattered',
    'sunroof exploded shattered',
    'sunroof spontaneously shattered',
    'sun roof shattered',
    'rear window exploded',
    'roof shattered',
    'windshield crack',
    'windshield cracked',
    'cracked windshield',
    'crack windshield',

    # Airbags/restraints
    'airbags deployed',
    'passenger airbag light',
    'seat belt',
    'seat belts',

    # Tires
    'tire blew',
    'tire failure',

    # Drivetrain/fuel
    'drive train malfunction',
    'fuel pump failed',
    'fuel leaking',
    'transmission issues',

    'bluetooth',
    'lane departure steering',
    'backup assist disengaged',
    'phantom braking',
    'lane assist active',
    'camera',
    'pre sense braking',
    'warning braking engaged',
    'false crash',

    'emergency braking activated',
    'emergency braking slammed', 
    'automatic emergency braking',
    'braking failed',
    'braking forward collision',
    'unsafe brakes sudden',
    'hard brake',

    # Steering — in BERT output, missing from your list  
    'steering issues',
    'power steering malfunctioned',

    # Transmission/drivetrain — in BERT output, weak in your list
    'transmission dying',
    'transmission died',
    'defect transmission',

    # Restraints — in BERT output, missing
    'seat belt warning',
    'safety restraint warning',
    'seat belts fail',
    'seat belts lock',
    'seat belts malfunctioned',
    'auto seat belt',

    # ADAS — in BERT output, missing
    'falsely triggered crash',
    'false crash event',
    'emergency braking aeb',
    'braking failed respond',
    'collision jolt',

    # Camera/visibility — in BERT output, missing
    'rear camera',
    'rearview camera',
    'reverse rearview camera',
    'camera temporarily unavailable',

    # Glazing — in BERT output, missing
    'sunroof popped shattered',
    'sunroof glass shattered',
    'sunroof exploded randomly',
    'rear windshield',

    'electrical fire',
    'burning smell electrical',
    'wiring harness',
    'electrical short',
    'power loss driving',

    # Engine — very thin in both lists
    'engine stalls',
    'engine shuts off',
    'engine cuts out',
    'sudden loss power',
    'engine seized',

    # EV/hybrid — absent everywhere
    'battery fire',
    'charging failure',
    'thermal runaway',

        # Electrical — strong additions
    'electrical malfunction',
    'electrical malfunctions',
    'overheating electrical failures',
    'examined overheating electrical',
    'battery significant hazard',
    'battery fails driving',
    'battery fails',
    'battery failing',
    'bluetooth safety hazard',  # keep unlike plain 'bluetooth'
    'ev unable restart',
    'trailer plug melted',
    'burned driving trailer',

    # Engine
    'engine overheating',
    'check engine light',  # debatable but high volume signal
    'oil leak',
    'timing chain snapped',
    'coolant leaked engine',
    'engine misfiring',
    'low oil warning',
    'oil consumption issue',
    'turbo fails',
    'ignition coil failed',
    'failing fuel engine',

    # Powertrain/transmission
    'transmission dying',
    'transmission issues',
    'transmission failed',
    'transmission jerks decelerating',
    'sudden loss drive',
    'delay transmission engaging',
    'transmission engages delay',
    'feel transmission dropping',
    'dct shudders sluggish',
    'transmission transfer case',
    'leak transfer case',

    # ADAS/collision
    'collision warning activates',
    'false alarms powertrain',
    'emergency braking',
    'issue adaptive cruise',

    # EV specific
    'turtle mode error',  # EV power limitation warning
    'fully charged danger',
    'failing battery thermostat',


]

SAFETY_KEYPHRASES = list(set(SAFETY_KEYPHRASES))

print("here")


import re

def keybert_safety_score(text):
    if not text:
        return 0
    text = str(text).lower()
    tokens = set(re.findall(r'\b\w+\b', text))
    print(tokens)
    
    score = 0
    for phrase in SAFETY_KEYPHRASES:
        phrase_tokens = set(phrase.lower().split())
        print(phrase_tokens)
        overlap = len(phrase_tokens & tokens) / len(phrase_tokens)
        if overlap >= 0.75:
            score += 1
    
    # normalize by description length
    return score


def vehicle_keybert_features(complaint_texts):
    """complaint_texts is a list of individual complaint descriptions"""
    scores = [keybert_safety_score(t) for t in complaint_texts]
    scores = [s for s in scores if s > 0]  # ignore zero-score complaints
    
    if not scores:
        return pd.Series({
            'keybert_mean': 0,
            'keybert_max': 0,
            'keybert_severe_count': 0,  # complaints scoring >= 8
            'keybert_severe_ratio': 0,  # what fraction of complaints are severe
        })
    
    return pd.Series({
        'keybert_mean': np.mean(scores),
        'keybert_max': max(scores),
        'keybert_severe_count': sum(s >= 8 for s in scores),
        'keybert_severe_ratio': sum(s >= 8 for s in scores) / len(complaint_texts),
    })

'''
# Apply per vehicle
vehicle_features = (
    complaint_data_combined
    .groupby(['MAKE', 'MODEL', 'YEAR'])['CDESCR']
    .apply(list)
    .apply(vehicle_keybert_features)
    .reset_index()
)
'''

complaint_data_combined = complaint_data_combined.groupby(["MAKE", "MODEL", "YEAR"]).agg(
    complaint_count = ("ODINO", "count"),
    crash_count = ("CRASH", "sum"),
    fire_count = ("FIRE", "sum"),  
    injured_count = ("INJURED", "sum"),
    death_count = ("DEATHS", "sum"),
    medical_count = ("MEDICAL_ATTN", "sum"),
    tow_count = ("VEHICLES_TOWED_YN", "sum"),
    police_count = ("POLICE_RPT_YN", "sum"),
    first_complaint_date = ("FAILDATE", "min"),
    last_complaint_date = ("FAILDATE", "max"),
    complaints_first_12m = ('within_12_months', 'sum'),
    median_mileage = ("MILES", "median"),
    component = ("COMPDESC", lambda x: list(x.dropna())),
    days_to_peak=("LDATE", days_to_peak),
    description = ("CDESCR", lambda x: " ".join(x.dropna()))).reset_index()

complaint_data_combined['crash_ratio'] = complaint_data_combined['crash_count'] / complaint_data_combined['complaint_count']
complaint_data_combined['fire_ratio'] = complaint_data_combined['fire_count'] / complaint_data_combined['complaint_count']
complaint_data_combined['injured_ratio'] = complaint_data_combined['injured_count'] / complaint_data_combined['complaint_count']
complaint_data_combined['death_ratio'] = complaint_data_combined['death_count'] / complaint_data_combined['complaint_count']

complaint_data_combined['medical_ratio'] = complaint_data_combined['medical_count'] / complaint_data_combined['complaint_count']
complaint_data_combined['tow_ratio'] = complaint_data_combined['tow_count'] / complaint_data_combined['complaint_count']
complaint_data_combined['police_ratio'] = complaint_data_combined['police_count'] / complaint_data_combined['complaint_count']


complaint_data_combined["days_span"] = (complaint_data_combined["last_complaint_date"] - complaint_data_combined["first_complaint_date"]).dt.days
complaint_data_combined["complaints_per_day"] = complaint_data_combined["complaint_count"] / complaint_data_combined["days_span"]
complaint_data_combined = complaint_data_combined.replace([np.inf, -np.inf], 0)
complaint_data_combined['median_mileage'] = complaint_data_combined['median_mileage'].fillna(0)

investigation = pd.read_csv("investigations.csv")
investigation['under_investigation'] = 1
investigation = investigation.drop_duplicates(subset = ["MAKE", "MODEL", "YEAR"])

complaint_data_combined = complaint_data_combined.merge(investigation[['MAKE', 'MODEL', 'YEAR', 'under_investigation']], on = ['MAKE', 'MODEL', 'YEAR'], how = 'left')

complaint_data_combined["under_investigation"] = pd.to_numeric(complaint_data_combined["under_investigation"], errors="coerce")
complaint_data_combined['under_investigation'] = complaint_data_combined['under_investigation'].fillna(0)
complaint_data_combined["under_investigation"] = complaint_data_combined["under_investigation"].astype(int)

#complaint_data_combined = complaint_data_combined.merge(vehicle_features[['MAKE', 'MODEL', 'YEAR', 'keybert_max', 'keybert_mean']], on = ['MAKE', 'MODEL', 'YEAR'], how = 'left')


complaint_data_combined['early_complaint_ratio'] = (
    complaint_data_combined['complaints_first_12m'] / 
    complaint_data_combined['complaint_count']
).fillna(0)

complaint_data_combined = complaint_data_combined.loc[complaint_data_combined['complaint_count'] >= 10]


PASSENGER_MAKES = {
    "ACURA", "AUDI", "BMW",
    "BUICK", "CADILLAC", "CHEVROLET", "CHRYSLER", "DODGE", 
    "FORD", "GMC", "HONDA", "HUMMER", "HYUNDAI",
    "INFINITI", "JAGUAR", "JEEP", "KIA", "LAND ROVER",
    "LEXUS", "LINCOLN", "MAZDA",
    "MERCEDES-BENZ", "MERCEDES", "MINI",
    "MITSUBISHI", "NISSAN", "POLESTAR",
    "PORSCHE", "RAM",
    "SUBARU", "TESLA", "TOYOTA", "VOLKSWAGEN", "VOLVO"
}

recall_data_combined = recall_data_combined[recall_data_combined["MAKE"].str.upper().isin(PASSENGER_MAKES)]
recall_data_combined = recall_data_combined.drop_duplicates(subset = ["MAKE", "MODEL", "YEAR"], keep='last')
recall_data_combined = recall_data_combined.rename(columns = {"MODEL YEAR": "YEAR"})
recall_data_combined['Recall'] = 1

recall_data_combined_year = recall_data_combined[(recall_data_combined["YEAR"] >= 2010) & (recall_data_combined["YEAR"] <= 2020)]
recall_data_combined_year['Recall'] = 1

recall_data_combined_year = recall_data_combined_year.rename(columns = {"MODEL YEAR": "YEAR"})

complaint_data_combined_year = complaint_data_combined[(complaint_data_combined["YEAR"] >= 2010) & (complaint_data_combined["YEAR"] <= 2020)]

complaint_recall = complaint_data_combined_year.merge(recall_data_combined_year[['MAKE', 'MODEL', 'YEAR', 'Recall']], on = ['MAKE', 'MODEL', 'YEAR'], how = 'left')
complaint_recall["Recall"] = pd.to_numeric(complaint_recall["Recall"], errors="coerce")
complaint_recall["Recall"] = complaint_recall["Recall"].fillna(0)
complaint_recall["Recall"] = complaint_recall["Recall"].astype(int)

manufacturer_stats = complaint_recall.groupby("MAKE").agg(
    total_models=("Recall", "count"),
    total_recalled=("Recall", "sum")
).reset_index()

manufacturer_stats["manufacturer_recall_rate"] = (
    manufacturer_stats["total_recalled"] / manufacturer_stats["total_models"]
)

print(manufacturer_stats.sort_values("manufacturer_recall_rate", ascending=False))

complaint_recall = complaint_recall.merge(
    manufacturer_stats[["MAKE", "manufacturer_recall_rate"]],
    on="MAKE",
    how="left"
)

print("here")

complaint_recall['has_steering'] = complaint_recall['component'].apply(
    lambda x: int(any('STEERING' in str(c) for c in x)) if x else 0
)

print("here")




'''
complain_data_combined_scoring = complain_data_combined_scoring.merge(complaint_recall[['MAKE', 'MODEL', 'YEAR', 'Recall']], on = ['MAKE', 'MODEL', 'YEAR'], how = 'left')
df = complain_data_combined_scoring
df = complain_data_combined_scoring['COMPDESC'].value_counts()
df.to_csv("test.csv")
'''

'''
from keybert import KeyBERT
import pandas as pd


import pandas as pd
from keybert import KeyBERT
from tqdm import tqdm

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

complaint_data_combined_test = complaint_data_combined[(complaint_data_combined["YEAR"] >= 2020) & (complaint_data_combined["YEAR"] <= 2026)]
recall_data_combined_test = recall_data_combined[(recall_data_combined["YEAR"] >= 2020) & (recall_data_combined["YEAR"] <= 2026)]
recall_data_combined_test['Recall'] = 1



recall_data_combined_test = recall_data_combined_test.rename(columns = {"MODEL YEAR": "YEAR"})

complaint_recall_test = complaint_data_combined_test.merge(recall_data_combined_test[['MAKE', 'MODEL', 'YEAR', 'Recall']], on = ['MAKE', 'MODEL', 'YEAR'], how = 'left')

complaint_recall_test["summary_clean"] = complaint_recall_test["description"].apply(clean_text)

miclassified = pd.read_excel("misclassified.xlsx")

complaint_recall_test = complaint_recall_test.merge(miclassified[['MAKE', 'MODEL', 'YEAR']], on = ['MAKE', 'MODEL', 'YEAR'], how = 'inner')

print(complaint_recall_test)

recall_df = complaint_recall_test

recall_df['short_text'] = recall_df['summary_clean'].str.slice(0, 2000)

results = []

docs = recall_df['summary_clean'].str.slice(0, 500).tolist()

print("Starting extraction...")
for text in tqdm(recall_df['short_text']):
    try:
        kp = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english', 
            top_n=5
        )
        # Just keep the words, discard the scores for speed
        words = [val[0] for val in kp]
        results.append(", ".join(words))
    except:
        results.append("Error in processing")

recall_df['top_words'] = results
print("Done!")
print(recall_df[['top_words']].head())

recall_df.to_csv("words.csv")
'''

print("here")



import re

def keybert_safety_score(text):
    if not text:
        return 0
    text = str(text).lower()
    tokens = set(re.findall(r'\b\w+\b', text))
    
    score = 0
    for phrase in SAFETY_KEYPHRASES:
        phrase_tokens = set(phrase.lower().split())
        overlap = len(phrase_tokens & tokens) / len(phrase_tokens)
        if overlap >= 0.75:
            score += 1
    return score

complaint_recall["keybert_safety_score"] = complaint_recall["description"].apply(keybert_safety_score)
complaint_recall["keybert_per_complaint"] = complaint_recall["keybert_safety_score"] / complaint_recall["complaint_count"]

def component_score(comp_list):
    if not comp_list:
        return 0
    text = ' '.join(str(c) for c in comp_list)
    count = sum(1 for phrase in COMPONENT_LIST if phrase in text)
    return count

complaint_recall["component_score"] = complaint_recall["component"].apply(component_score)

print("here")


score = complaint_recall[['MAKE', 'MODEL', 'YEAR', 'component_score']]

print(complaint_recall.groupby("Recall")[["keybert_safety_score"]].median())

print(complaint_recall.groupby('Recall')['manufacturer_recall_rate'].mean())

print(complaint_recall.groupby('Recall')['under_investigation'].mean())

#print(complaint_recall.groupby("Recall")[["keybert_max"]].median())
#print(complaint_recall.groupby("Recall")[["keybert_mean"]].median())


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import shap


features = [
    "crash_ratio",
    'keybert_safety_score',
    "median_mileage",
    "complaints_first_12m",
    "tow_ratio"
]


# time series split — train on older, test on newer

from sklearn.model_selection import train_test_split

X = complaint_recall[features]
y = complaint_recall["Recall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # ensures same class ratio in train and test
)


print(f"Train recall rate: {y_train.mean():.2%}")
print(f"Test recall rate: {y_test.mean():.2%}")

from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(class_weight = class_weight_dict)
lr.fit(X_train_scaled, y_train)


rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, roc_auc_score

scoring = {
    'recall':    make_scorer(recall_score),
    'precision': make_scorer(precision_score),
    'f1':        make_scorer(f1_score),
    'roc_auc':   make_scorer(roc_auc_score)
}

# Stratified 5-fold — preserves your 70/30 class ratio in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run for each model
for name, model in [("XGBoost", xgb), 
                     ("Logistic Regression", lr),
                     ("Random Forest", rf)]:
    
    results = cross_validate(model, X_train, y_train, 
                             cv=cv, scoring=scoring)
    
    print(f"\n{name} Cross Validation (5-fold):")
    print(f"  Recall:    {results['test_recall'].mean():.3f} ± {results['test_recall'].std():.3f}")
    print(f"  Precision: {results['test_precision'].mean():.3f} ± {results['test_precision'].std():.3f}")
    print(f"  F1:        {results['test_f1'].mean():.3f} ± {results['test_f1'].std():.3f}")
    print(f"  AUC:       {results['test_roc_auc'].mean():.3f} ± {results['test_roc_auc'].std():.3f}")



lr_preds = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
print("\nLogistic Regression:")
print(classification_report(y_test, lr_preds))
print(f"AUC: {roc_auc_score(y_test, lr_proba):.3f}")

for threshold in [0.35, 0.40, 0.475, 0.50]:
    y_pred = (lr_proba >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print(classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, lr_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
print("\nRandom Forest:")
print(classification_report(y_test, rf_preds))
print(f"AUC: {roc_auc_score(y_test, rf_proba):.3f}")

fpr, tpr, thresholds = roc_curve(y_test, rf_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

print("\nXGBoost:")
print(classification_report(y_test, xgb_preds))
print(f"AUC: {roc_auc_score(y_test, xgb_proba):.3f}")

fpr, tpr, thresholds = roc_curve(y_test, xgb_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

fn_mask = (y_test == 1) & (xgb_preds == 0)
fn_cases = X_test[fn_mask].copy()
fn_cases['true_label'] = y_test[fn_mask]

from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
import pandas as pd

# Binary correct/incorrect for each model
xgb_correct = (xgb_preds == y_test)
lr_correct = (lr_preds == y_test)
rf_correct = (rf_preds == y_test)

def run_mcnemar(correct_a, correct_b, name_a, name_b):
    # Contingency table
    # b = A correct, B wrong
    # c = A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    
    print(f"\n{name_a} vs {name_b}")
    print(f"  {name_a} right, {name_b} wrong: {b}")
    print(f"  {name_a} wrong, {name_b} right: {c}")
    print(f"  p-value: {result.pvalue:.4f}")
    print(f"  Significant: {'Yes' if result.pvalue < 0.05 else 'No'}")

# Compare all pairs
run_mcnemar(xgb_correct, lr_correct, "XGBoost", "Logistic Regression")
run_mcnemar(xgb_correct, rf_correct, "XGBoost", "Random Forest")
run_mcnemar(lr_correct,  rf_correct, "Logistic Regression", "Random Forest")



from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV

param_grid = {'C': np.logspace(-4, 4, 10)}

# Run the grid search
grid = GridSearchCV(
    LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000),
    param_grid,
    cv=5,
    scoring='roc_auc'
)
grid.fit(X_train, y_train)

print(f"Best C value: {grid.best_params_['C']}")

# evaluate best model on test set
best_xgb = grid.best_estimator_
xgb_preds = best_xgb.predict(X_test_scaled)
xgb_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]

lower_threshold = 0.5
y_pred_high_recall = (xgb_proba >= lower_threshold).astype(int)

print("\nTuned XGBoost:")
print(classification_report(y_test, y_pred_high_recall))
print(f"Test AUC: {roc_auc_score(y_test, y_pred_high_recall):.3f}")

print(f"Recalled as 1: {(y_pred_high_recall == 1).sum()}")
print(f"Recalled as 1: {(xgb_preds == 1).sum()}")


recall_data_combined_test = recall_data_combined[(recall_data_combined["YEAR"] >= 2022) & (recall_data_combined["YEAR"] <= 2026)]
recall_data_combined_test = pd.read_csv("recalled.csv")
recall_data_combined_test['Recall'] = 1

recall_data_combined_test = recall_data_combined_test.rename(columns = {"MODEL YEAR": "YEAR"})

complaint_data_combined_test = complaint_data_combined[(complaint_data_combined["YEAR"] >= 2022) & (complaint_data_combined["YEAR"] <= 2026)]

complaint_recall_test = complaint_data_combined_test.merge(recall_data_combined_test[['MAKE', 'MODEL', 'YEAR', 'Recall']], on = ['MAKE', 'MODEL', 'YEAR'], how = 'left')
complaint_recall_test["Recall"] = pd.to_numeric(complaint_recall_test["Recall"], errors="coerce")
complaint_recall_test["Recall"] = complaint_recall_test["Recall"].fillna(0)
complaint_recall_test["Recall"] = complaint_recall_test["Recall"].astype(int)

complaint_recall_test = complaint_recall_test.merge(
    manufacturer_stats[["MAKE", "manufacturer_recall_rate"]],
    on="MAKE",
    how="left"
)

print(complaint_recall_test[complaint_recall_test.isna().any(axis=1)])


model_name = {
"MDX": "MDX TYPE S",

    
}



def keybert_safety_score(text):
    if not text:
        return 0
    text = str(text).lower()
    tokens = set(re.findall(r'\b\w+\b', text))
    
    score = 0
    for phrase in SAFETY_KEYPHRASES:
        phrase_tokens = set(phrase.lower().split())
        overlap = len(phrase_tokens & tokens) / len(phrase_tokens)
        if overlap >= 0.75:
            score += 1
    return score

complaint_recall_test["keybert_safety_score"] = complaint_recall_test["description"].apply(keybert_safety_score)
complaint_recall_test["keybert_per_complaint"] = complaint_recall_test["keybert_safety_score"] / complaint_recall_test["complaint_count"]

print(complaint_recall_test.groupby("Recall")[["keybert_safety_score"]].median())

base_rate = complaint_recall_test['Recall'].mean()
results = []
for phrase in set(SAFETY_KEYPHRASES):
    has_phrase = complaint_recall_test['description'].str.contains(phrase, case=False, na=False)
    n = has_phrase.sum()
    if n < 5:
        continue
    lift = complaint_recall_test[has_phrase]['Recall'].mean() / base_rate
    results.append((phrase, n, round(lift, 2)))

# Sort by lift ascending — bottom of this list are your noise phrases
results.sort(key=lambda x: x[1])
for phrase, n, lift in results:
    flag = ' REMOVE' if lift < 1.0 else ''
    print(f"{phrase:<45} n={n:>5}  lift={lift:.2f}x{flag}")


import joblib

joblib.dump(lr, 'lr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
lr_model_loaded = joblib.load('lr_model.pkl')


X_final = complaint_recall_test[features]
y_final = complaint_recall_test["Recall"]

X_test_scaled = scaler.transform(X_final)

prob_recall = lr_model_loaded.predict_proba(X_test_scaled)[:, 1]
chosen_threshold = 0.5
y_pred_custom = (prob_recall >= chosen_threshold).astype(int)

complaint_recall_test["Predict_Recall"] = y_pred_custom
complaint_recall_test["Probability_Recall"] = prob_recall

from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_final, y_pred_custom))
print(f"AUC: {roc_auc_score(y_final, prob_recall):.3f}")

import shap
import matplotlib.pyplot as plt

explainer = shap.LinearExplainer(lr_model_loaded, X_train_scaled)
shap_values = explainer(X_test_scaled)

# bar plot
shap.summary_plot(
    shap_values.values, 
    X_test_scaled,
    feature_names=features,
    plot_type="bar"
)
plt.savefig("shap_xgb_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# dot plot
shap.summary_plot(
    shap_values.values, 
    X_test_scaled,
    feature_names=features,
    show=False
)
plt.xlim(-5, 5)
plt.savefig("shap_xgb_dot.png", dpi=150, bbox_inches="tight")
plt.show()

shap_df = pd.DataFrame(
    shap_values.values,  
    columns = [f'shap_{col}' for col in features]
)

result = complaint_recall_test[[
    "MAKE", "MODEL", "YEAR",
    "Recall", "Predict_Recall", "Probability_Recall", "complaint_count",  "crash_ratio",
    "keybert_safety_score",
    "median_mileage",
    "fire_ratio",
    "complaints_first_12m",
    "under_investigation"]]

result.to_csv("result.csv")

for col in shap_df.columns:
    result[f'shap_{col}'] = shap_df[col]

yearly_stats = result.groupby('YEAR').apply(lambda g: pd.Series({
    'total_recalls': g['Recall'].sum(),
    'caught': ((g['Recall']==1) & (g['Predict_Recall']==1)).sum(),
    'missed': ((g['Recall']==1) & (g['Predict_Recall']==0)).sum(),
    'sensitivity': ((g['Recall']==1) & (g['Predict_Recall']==1)).sum() / max(g['Recall'].sum(), 1),
    'avg_keybert': g['keybert_safety_score'].mean(),
    'avg_complaints': g['complaints_first_12m'].mean(),
})).round(3)

print(yearly_stats)



predicted_recalled = complaint_recall_test[
    complaint_recall_test['Predict_Recall'] == 1
]

print("Probability distribution for predicted recalls:")
print(predicted_recalled['Probability_Recall'].describe())

# Bucket by certainty
very_high = predicted_recalled[predicted_recalled['Probability_Recall'] >= 0.85]
high = predicted_recalled[(predicted_recalled['Probability_Recall'] >= 0.70) & 
                           (predicted_recalled['Probability_Recall'] < 0.85)]
medium = predicted_recalled[(predicted_recalled['Probability_Recall'] >= 0.55) & 
                             (predicted_recalled['Probability_Recall'] < 0.70)]
low = predicted_recalled[predicted_recalled['Probability_Recall'] < 0.55]

print(f"\nVery high certainty (>=85%): {len(very_high)}")
print(f"High certainty (70-85%): {len(high)}")
print(f"Medium certainty (55-70%): {len(medium)}")
print(f"Low certainty (threshold-55%): {len(low)}")

# Actual recall rate within each certainty bucket
for label, subset in [('Very high', very_high), ('High', high), 
                       ('Medium', medium), ('Low', low)]:
    if len(subset) > 0:
        actual_recall_rate = subset['Recall'].mean()
        print(f"{label} certainty actual recall rate: {actual_recall_rate:.2%}")


result['prob_band'] = pd.cut(result['Probability_Recall'], 
                          bins=[0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          labels=['<0.3','0.3-0.5','0.5-0.6','0.6-0.7',
                                  '0.7-0.8','0.8-0.9','0.9-1.0'])

summary = result.groupby('prob_band').agg(
    total=('Recall','count'),
    actual_recalls=('Recall','sum'),
    recall_rate=('Recall','mean')
).round(3)
print(summary)

'''
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

complaint_recall["summary_clean"] = complaint_recall["description"].apply(clean_text)



NOISE_PATTERNS = [
    r'\b\d{4,5}\b',           # zip codes and years
    r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',  # months
    r'\b(acura|ford|toyota|honda|bmw|jeep|kia)\b',  # brand names
    r'\b(vehicle|car|automobile|cars|vehicles)\b',  # generic nouns
    r'\b(va|nj|ca|tx|fl|ny)\b',  # state codes
]

def is_noise(phrase):
    phrase = phrase.lower()
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, phrase):
            return True
    return False

def filter_keyphrases(phrases):
    return [p for p in phrases if not is_noise(p)]



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

from keybert import KeyBERT
import pandas as pd


import pandas as pd
from keybert import KeyBERT
from tqdm import tqdm

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

recall_df = complaint_recall[complaint_recall['Recall'] == 1].copy()

recall_df['short_text'] = recall_df['summary_clean'].str.slice(0, 2000)

results = []

docs = recall_df['summary_clean'].str.slice(0, 500).tolist()

print("Starting extraction...")
for text in tqdm(recall_df['short_text']):
    try:
        # Extract keywords for one car at a time
        kp = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english', 
            top_n=5
        )
        # Just keep the words, discard the scores for speed
        words = [val[0] for val in kp]
        results.append(", ".join(words))
    except:
        results.append("Error in processing")

recall_df['top_words'] = results
print("Done!")
print(recall_df[['top_words']].head())



from collections import Counter


def parse_phrases(text):
    if not isinstance(text, str):
        return []
    return [p.strip() for p in text.split(",")]

recall_df["top_words"] = recall_df["top_words"].apply(parse_phrases)


NOISE_PATTERNS = [
    r'\b\d{4,5}\b',           # zip codes and years
    r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',  # months
    r'\b(acura|ford|toyota|honda|bmw|jeep|kia)\b',  # brand names
    r'\b(vehicle|car|automobile|cars|vehicles)\b',  # generic nouns
    r'\b(va|nj|ca|tx|fl|ny)\b',  # state codes
]

def is_noise(phrase):
    phrase = phrase.lower()
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, phrase):
            return True
    return False

def filter_keyphrases(phrases):
    return [p for p in phrases if not is_noise(p)]


recall_df["top_words_filtered"] = recall_df["top_words"].apply(filter_keyphrases)


# step 1: flatten all keyphrases across all vehicles into one list
all_phrases = [
    phrase
    for phrases in recall_df["top_words_filtered"]
    for phrase in phrases
]

# step 2: count most common
phrase_counts = Counter(all_phrases)
top_phrases = [phrase for phrase, count in phrase_counts.most_common(50)]
print(top_phrases)


# how many unique vehicles contributed to these phrases
print(f"Total vehicles: {len(recall_df)}")

# check phrase distribution across components
print(recall_df["top_words"].apply(lambda x: 
    any("brake" in p for p in x)).sum(), "vehicles with brake phrases")
print(recall_df["top_words"].apply(lambda x: 
    any("engine" in p for p in x)).sum(), "vehicles with engine phrases")
print(recall_df["top_words"].apply(lambda x: 
    any("airbag" in p for p in x)).sum(), "vehicles with airbag phrases")
print(recall_df["top_words"].apply(lambda x: 
    any("steer" in p for p in x)).sum(), "vehicles with steering phrases")



SAFETY_PHRASES = [
    "brake", "stall", "fire", "smoke", "accelerat",
    "rollover", "collision", "crash", "unsafe", "overheat",
    "airbag", "air bag", "locked", "ceased", "failed",
    "sudden", "without warning", "lost control"
]

def keybert_safety_score(phrases):
    if not phrases:
        return 0
    score = sum(
        1 for phrase in phrases
        if any(safety in phrase.lower() for safety in SAFETY_PHRASES)
    )
    return score / len(phrases)

recall_df["top_words_filtered"] = recall_df["top_words"].apply(filter_keyphrases)

# step 3: score based on filtered phrases
recall_df["keybert_safety_score"] = recall_df["top_words_filtered"].apply(keybert_safety_score)



# sanity check
print(recall_df.groupby("Recall")[["keybert_safety_score"]].mean())


# fit TF-IDF on complaint summaries
tfidf = TfidfVectorizer(
    max_features=100,    # top 100 words
    ngram_range=(1, 2),  # single words and two-word phrases
    stop_words="english",
    min_df=5             # word must appear in at least 5 vehicles
)

X_tfidf = tfidf.fit_transform(final_df["summary_clean"])

# quick logistic regression to find which words predict recalls
lr = LogisticRegression(class_weight="balanced")
lr.fit(X_tfidf, final_df["Recall"])

# see top words associated with recall
feature_names = tfidf.get_feature_names_out()
coefficients = lr.coef_[0]

word_importance = pd.DataFrame({
    "word": feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", ascending=False)

print("Top 20 words predicting RECALL:")
print(word_importance.head(20))

print("\nTop 20 words predicting NO RECALL:")
print(word_importance.tail(20))

import spacy
from keybert import KeyBERT
import pandas as pd
from tqdm import tqdm




# 1. Initialize with a specific limit to prevent hanging
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# 2. Prep your data
# Assuming df is your grouped dataframe
recall_df = complaint_recall[complaint_recall['Recall'] == 1].copy()

# TRUNCATE text: Transformers can't read more than ~400-500 words anyway.
# This prevents the "stuck" behavior.
recall_df['short_text'] = recall_df['description'].str.slice(0, 2000)

results = []

print("Starting extraction...")
# 3. Loop with a progress bar so you know it's moving
for text in tqdm(recall_df['short_text']):
    try:
        # Extract keywords for one car at a time
        kp = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english', 
            top_n=5
        )
        # Just keep the words, discard the scores for speed
        words = [val[0] for val in kp]
        results.append(", ".join(words))
    except:
        results.append("Error in processing")

recall_df['top_words'] = results
print("Done!")
print(recall_df['top_words'])
print(recall_df[['Model', 'top_words']].head())


# 5. Global Summary (The "Top Words" across all recalled cars)
all_recall_text = " ".join(recall_df['cleaned_text'])
global_top_words = kw_model.extract_keywords(all_recall_text, top_n=10)

print("--- Global Top Recall Indicators ---")
print(global_top_words)

print("\n--- Per Car Sample ---")
print(recall_df[['car_model', 'top_recall_phrases']].head())'''
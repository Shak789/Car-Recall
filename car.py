import pandas as pd
import re


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
    'steering fault',
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
    'steering fault',

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
    'false crash'

]

complaint_data_10_14 = pd.read_csv('Complaint_2010_2014.csv', header = None, names = nhtsa_headers)
complaint_data_15_19 = pd.read_csv('Complaint_2015_2019.csv', header = None, names = nhtsa_headers)
complaint_data_20_24 = pd.read_csv('Complaint_2020_2024.csv', header = None, names = nhtsa_headers)
complaint_data_25_26 = pd.read_csv('Complaint_2025_2026.csv', header = None, names = nhtsa_headers)

complaint_data_combined = pd.concat([complaint_data_10_14, complaint_data_15_19, complaint_data_20_24, complaint_data_25_26])



def keybert_debug(text, top_n=5):
    if not text:
        return []
    text = str(text).lower()
    tokens = set(re.findall(r'\b\w+\b', text))
    
    matches = []
    for phrase in SAFETY_KEYPHRASES:
        phrase_tokens = set(phrase.lower().split())
        overlap = len(phrase_tokens & tokens) / len(phrase_tokens)
        if overlap >= 0.75:
            matches.append(phrase)
    return matches

misclassified = pd.read_excel("misclassified.xlsx")

complaint_data_combined_test = complaint_data_combined[(complaint_data_combined["YEAR"] >= 2020) & (complaint_data_combined["YEAR"] <= 2026)]

fn_df = complaint_data_combined_test.merge(misclassified[['MAKE', 'MODEL', 'YEAR']], on = ['MAKE', 'MODEL', 'YEAR'], how = 'inner')


# Apply to your missed recalls
fn_df['matched_phrases'] = fn_df['description'].apply(keybert_debug)
fn_df['match_count'] = fn_df['matched_phrases'].apply(len)

print(fn_df['match_count'].value_counts().sort_index())
print("\nSample missed recall with zero phrase matches:")
print(fn_df[fn_df['match_count'] == 0]['description'].iloc[0])
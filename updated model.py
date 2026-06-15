import pandas as pd
import numpy as np
import re

NHSTA_HEADERS = [
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

SAFETY_KEYPHRASE = [
    "air bags",
    "airbag",
    "airbags deployed",
    "auto seat belt",
    "auto stop",
    "auto stop start",
    "automatic emergency braking",
    "backup assist disengaged",
    "battery failing",
    "battery fails",
    "battery fails driving",
    "battery fire",
    "battery significant hazard",
    "bluetooth",
    "bluetooth safety hazard",
    "brake failure",
    "brake fluid",
    "braking failed",
    "braking failed respond",
    "braking forward collision",
    "burned driving trailer",
    "burning smell electrical",
    "camera",
    "camera temporarily unavailable",
    "charging failure",
    "check engine light",
    "collision jolt",
    "collision warning activates",
    "coolant leaked engine",
    "crack windshield",
    "cracked windshield",
    "dct shudders sluggish",
    "defect transmission",
    "delay transmission engaging",
    "drive train malfunction",
    "electrical fire",
    "electrical malfunction",
    "electrical malfunctions",
    "electrical short",
    "emergency brake",
    "emergency brake engaged",
    "emergency braking",
    "emergency braking activated",
    "emergency braking aeb",
    "emergency braking slammed",
    "engine cuts out",
    "engine misfiring",
    "engine overheating",
    "engine seized",
    "engine shuts off",
    "engine stalls",
    "ev unable restart",
    "examined overheating electrical",
    "failing battery thermostat",
    "failing fuel engine",
    "false alarms powertrain",
    "false crash",
    "false crash event",
    "falsely triggered crash",
    "feel transmission dropping",
    "fuel leaking",
    "fuel pump failed",
    "fully charged danger",
    "hard brake",
    "highway tire separated",
    "ignition coil failed",
    "issue adaptive cruise",
    "lane assist active",
    "lane departure steering",
    "leak transfer case",
    "lost power steering",
    "low oil warning",
    "oil consumption issue",
    "oil leak",
    "overheating electrical failures",
    "passenger airbag light",
    "phantom braking",
    "power loss driving",
    "power steering failed",
    "power steering malfunctioned",
    "pre sense braking",
    "rear camera",
    "rear window exploded",
    "rear windshield",
    "rearview camera",
    "reverse rearview camera",
    "roof shattered",
    "safety restraint warning",
    "seat belt",
    "seat belt warning",
    "seat belts",
    "seat belts fail",
    "seat belts lock",
    "seat belts malfunctioned",
    "steering failed",
    "steering issues",
    "steering stopped",
    "steering wheel locked",
    "steering wheel seized",
    "sudden loss drive",
    "sudden loss power",
    "sun roof shattered",
    "sunroof exploded",
    "sunroof exploded driving",
    "sunroof exploded randomly",
    "sunroof exploded shattered",
    "sunroof glass shattered",
    "sunroof popped shattered",
    "sunroof shattered",
    "sunroof spontaneously shattered",
    "thermal runaway",
    "timing chain snapped",
    "tire blew",
    "tire failure",
    "trailer plug melted",
    "transmission died",
    "transmission dying",
    "transmission engages delay",
    "transmission failed",
    "transmission issues",
    "transmission jerks decelerating",
    "transmission transfer case",
    "turbo fails",
    "turtle mode error",
    "unsafe brakes sudden",
    "warning braking engaged",
    "windshield crack",
    "windshield cracked",
    "wiring harness" 
]

complaint_data_10_14 = pd.read_csv('Complaint_2010_2014.csv', header = None, names = NHSTA_HEADERS)
complaint_data_15_19 = pd.read_csv('Complaint_2015_2019.csv', header = None, names = NHSTA_HEADERS)
complaint_data_20_24 = pd.read_csv('Complaint_2020_2024.csv', header = None, names = NHSTA_HEADERS)
complaint_data_25_26 = pd.read_csv('Complaint_2025_2026.csv', header = None, names = NHSTA_HEADERS)
complaint_data_combined = pd.concat([complaint_data_10_14, complaint_data_15_19, complaint_data_20_24, complaint_data_25_26])

complaint_data_combined["YEARTXT"] = pd.to_numeric(complaint_data_combined["YEARTXT"], errors="coerce")
complaint_data_combined = complaint_data_combined.dropna(subset=["YEARTXT"])
complaint_data_combined["YEARTXT"] = complaint_data_combined["YEARTXT"].astype(int)
complaint_data_combined = complaint_data_combined.loc[complaint_data_combined['YEARTXT'] != 9999]

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


complaint_data_combined['RELEASE_DATE'] = pd.to_datetime(
    complaint_data_combined['YEAR'].astype(str) + '-01-01'
)

complaint_data_combined['within_12_months'] = (
    (complaint_data_combined['LDATE'] - complaint_data_combined['RELEASE_DATE']).dt.days.between(0, 365)
)


recall_data_combined = pd.read_csv('recalled_2010_2020.csv')

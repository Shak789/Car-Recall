import pandas as pd
import json
import ollama  # pip install ollama -- also need ollama installed locally: https://ollama.com

test = pd.read_csv("test.csv")


# pull model first time: ollama pull llama3

# ---replace these with your actual data---
# strip make/model/year from complaint text before sending



mask = (
    (test["MAKETXT"] == "BUICK") &
    (test["MODELTXT"] == "ENCLAVE") &
    (test["YEARTXT"] == 2025)
)

sample_complaints = (
    test[mask]["CDESCR"]
    .dropna()
    .sample(min(20, mask.sum()), random_state=42)
    .tolist()
)

print(f"Sampled {len(sample_complaints)} complaints")
print(sample_complaints[0])  


prompt = f"""You are assessing vehicle complaints filed with the National Highway Traffic Safety Administration (NHTSA).

A recall is issued when a vehicle creates an UNREASONABLE SAFETY RISK or fails to meet minimum safety standards.
Unreasonable safety risks include: sudden loss of control, brake failure, engine stalls at speed, fire risk,
airbag malfunction, steering failure, fuel leaks, electrical fires, and similar defects that could cause
injury or death.

This does NOT include: infotainment bugs, bluetooth issues, cosmetic defects,
minor rattles, noise complaints (humming, rattling, vibration without loss of 
control), backup camera glitches, fluid leaks found during routine inspection 
with no fire or loss-of-control consequence, steering vibration without loss 
of vehicle control, or general quality complaints with no safety consequence.


Below are {len(sample_complaints)} complaints for an anonymous vehicle. For each complaint,
determine if it describes an unreasonable safety risk (YES or NO).


Respond ONLY with a JSON object in this exact format, no preamble, no explanation, no markdown:
{{
    "assessments": [
        {{"complaint_number": 1, "is_safety_risk": true, "reason": "one sentence"}},
        {{"complaint_number": 2, "is_safety_risk": false, "reason": "one sentence"}}
    ],
    "safety_risk_ratio": 0.6
}}

safety_risk_ratio is the proportion of complaints that are safety risks (0.0 to 1.0)."""

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}]
)

import re
import json

raw = response["message"]["content"].strip()
clean = raw.replace("```json", "").replace("```", "").strip()
clean = clean.replace(": YES", ": true").replace(": NO", ": false")

json_match = re.search(r'\{.*\}', clean, re.DOTALL)
if json_match:
    clean = json_match.group()

# remove any control characters that break json parsing
clean = re.sub(r'[\x00-\x1f\x7f]', ' ', clean)

# truncate reason fields if they're too long -- model sometimes dumps the full complaint in there
clean = re.sub(r'("reason":\s*")(.{0,150}?)(")', lambda m: m.group(1) + m.group(2) + m.group(3), clean)

result = json.loads(clean)

print(f"Safety Risk Ratio: {result['safety_risk_ratio']:.2f}\n")
for a in result["assessments"]:
    flag = "RISK" if a["is_safety_risk"] else "ok"
    print(f"[{flag}] Complaint {a['complaint_number']}: {a['reason']}")

import pandas as pd

card_df = pd.read_csv("card_db_raw.csv")
class_df = pd.read_csv("antibiotic_class_usage.csv")


CLASS_RISK = dict(zip(class_df["DrugClass"], class_df["ClassRiskScore"]))

def compute_arg_class_risk(drug_class_string):
    """
    Takes:
        "class1;class2;class3"
    
    Returns:
        MAX(class risk score)
    """
    
    if pd.isna(drug_class_string):
        return 0
    
    classes = drug_class_string.split(";")
    
    max_score = 0
    
    for cls in classes:
        cls = cls.strip()
        
        if cls in CLASS_RISK:
            score = CLASS_RISK[cls]
            
            if score > max_score:
                max_score = score
    
    return max_score


card_df["ComputedClassRisk"] = card_df["DrugClass"].apply(compute_arg_class_risk)

clinical_df = pd.read_csv("reference_clinically_relevant_args.csv")

card_df["ARG"] = card_df["ARG"].str.upper()
clinical_df["ARG"] = clinical_df["ARG"].str.upper()

CLINICAL_SET = set(clinical_df["ARG"])

def compute_mdr(class_string):
    """
    MDR = 1 if ≥3 UNIQUE classes, else 0
    """
    if pd.isna(class_string) or class_string == "":
        return 0

    classes = [c.strip() for c in class_string.split(";") if c.strip() != ""]
    
    unique_classes = set(classes)
    
    if len(unique_classes) >= 3:
        return 1
    else:
        return 0


def compute_arg_score(row):
    """
    ARG Score rule:
    - 4 if clinically relevant
    - else ComputedClassRisk
    """
    if row["ARG"] in CLINICAL_SET:
        return 4
    else:
        return row["ComputedClassRisk"]

#
card_df["MDR"] = card_df["DrugClass"].apply(compute_mdr)

card_df["ARG Score"] = card_df.apply(compute_arg_score, axis=1)

card_df["RiskSeverity"] = card_df["ARG Score"] + card_df["MDR"]


final_df = card_df[[
    "ARG",
    "DrugClass",
    "ARG Score",
    "MDR",
    "RiskSeverity"
]].rename(columns={
    "DrugClass": "Class"
})


# MGEs

mge_df = pd.read_csv("mges.csv")

ARG_TO_SCORE = dict(zip(card_df["ARG"], card_df["ARG Score"]))
ARG_TO_CLASS = dict(zip(card_df["ARG"], card_df["DrugClass"]))
VALID_ARGS = set(card_df["ARG"])

def extract_args(mge_string):
    """
    Split MGE and keep only valid ARGs
    """
    parts = str(mge_string).split("_")
    
    args = []
    
    for p in parts:
        p_clean = p.strip().upper()
        
        if p_clean in VALID_ARGS:
            args.append(p_clean)
    
    return args


def compute_mge_arg_score(arg_list):
    """
    Max ARG score across ARGs
    """
    if not arg_list:
        return 0
    
    scores = [ARG_TO_SCORE[arg] for arg in arg_list if arg in ARG_TO_SCORE]
    
    return max(scores) if scores else 0


def compute_mdr(arg_list):
    """
    MDR = 1 if ≥3 unique classes across ARGs
    """
    
    classes = []
    
    for arg in arg_list:
        if arg in ARG_TO_CLASS:
            class_string = ARG_TO_CLASS[arg]
            
            # split multi-class entries
            for c in class_string.split(";"):
                c_clean = c.strip().upper()
                if c_clean != "":
                    classes.append(c_clean)
    
    unique_classes = set(classes)
    
    return 1 if len(unique_classes) >= 3 else 0

results = []

for _, row in mge_df.iterrows():
    
    mge = row["MGE"]

    args = extract_args(mge)
    arg_score = compute_mge_arg_score(args)
    mdr = compute_mdr(args)
    risk_severity = arg_score + mdr
    
    results.append({
        "MGE": mge,
        "ARG score": arg_score,
        "MDR": mdr,
        "Risk Severity": risk_severity
    })


mge_df = pd.DataFrame(results)

# Plasmid Risks

sample_df = pd.read_csv("plasmid_samples.txt", sep="\t")

sample_df.columns = sample_df.columns.str.strip()

sample_df["Value"] = sample_df["Value"].astype(str).str.replace('"', '').str.strip()
sample_df["Attribute"] = sample_df["Attribute"].astype(str).str.replace('"', '').str.upper()


sample_df["ARG Score"] = sample_df["Attribute"].map(ARG_TO_SCORE)
sample_df["Class"] = sample_df["Attribute"].map(ARG_TO_CLASS)

sample_df = sample_df.dropna(subset=["ARG Score"])

results = []

for plasmid, group in sample_df.groupby("Value"):
    
    # --- ARG SCORE (MAX across samples) ---
    arg_score = group["ARG Score"].max()
    
    # --- MDR (across ALL samples) ---
    all_classes = []
    
    for cls in group["Class"].dropna():
        for c in str(cls).split(";"):
            c_clean = c.strip().upper()
            if c_clean != "":
                all_classes.append(c_clean)
    
    unique_classes = set(all_classes)
    
    mdr = 1 if len(unique_classes) >= 3 else 0
    
    # --- RISK SEVERIY ---
    risk_severity = arg_score + mdr
    
    results.append({
        "Plasmid": plasmid,
        "ARG score": int(arg_score),
        "MDR": mdr,
        "Risk Severity": int(risk_severity)
    })


plasmid_df = pd.DataFrame(results)

likelihood_df = pd.read_csv("likelihood_input.csv")
likelihood_df["Trait"] = likelihood_df["Trait"].str.upper()

# Likelihoods

def compute_likelihood(row):
    
    # --- PRESENCE (1 if >0) ---
    
    livestock_present = (
        (row["Chicken (farms)"] > 0) or 
        (row["Environment (farms)"] > 0)
    )
    
    food_present = (
        (row["Retail meat (Chicken or pork)"] > 0)
    )
    
    human_present = (
        (row["Food Workers"] > 0) or
        (row["Symptomatic humans"] > 0) or
        (row["Non-food Workers"] > 0)
    )
    
    # --- BASE LIKELIHOOD ---
    
    if livestock_present and not (food_present or human_present):
        L = 1
    
    elif food_present and not human_present:
        # food only OR food+livestock
        L = 2
    
    elif human_present and not (food_present or livestock_present):
        L = 3
    
    elif human_present and (food_present or livestock_present):
        if human_present and food_present and livestock_present:
            L = 5
        else:
            L = 4
    else:
        L = 1  # fallback (should not occur)
    
    # --- SEROVAR ADJUSTMENT ---
    
    if row["Serovars"] >= 2:
        L += 1
    
    return min(L, 5)


likelihood_df["Likelihood"] = likelihood_df.apply(compute_likelihood, axis=1)

# Combine

arg_df = final_df.rename(columns={
    "ARG": "Trait",
    "RiskSeverity": "Risk Severity"
})[["Trait", "Risk Severity"]]

valid_traits = set(likelihood_df["Trait"])
arg_df = arg_df[arg_df["Trait"].isin(valid_traits)]

mge_df = mge_df.rename(columns={
    "MGE": "Trait"
})[["Trait", "Risk Severity"]]

mge_df["Trait"] = mge_df["Trait"].str.upper()

plasmid_df = plasmid_df.rename(columns={
    "Plasmid": "Trait"
})[["Trait", "Risk Severity"]]

plasmid_df["Trait"] = plasmid_df["Trait"].str.upper()

severity_df = pd.concat([arg_df, mge_df, plasmid_df], ignore_index=True)


final_df = severity_df.merge(
    likelihood_df[["Trait", "Likelihood"]],
    on="Trait",
    how="left"
)

final_df = final_df.rename(columns={
    "Likelihood": "Risk Likelihood"
})

final_df["Risk Score"] = final_df["Risk Severity"] * final_df["Risk Likelihood"]

def categorize(score):
    if pd.isna(score):
        return "Unknown"
    elif 1 <= score <= 3:
        return "Rank I"
    elif 4 <= score <= 9:
        return "Rank II"
    elif 10 <= score <= 14:
        return "Rank III"
    elif 15 <= score <= 24:
        return "Rank IV"
    elif score == 25:
        return "Rank V"
    else:
        return "Unknown"

final_df["Risk Category"] = final_df["Risk Score"].apply(categorize)

final_df.to_csv("Final_RiskScores.csv", index=False)

print(final_df.head())

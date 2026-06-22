Risk Assessment Pipeline This Python script calculates risk scores for antimicrobial resistance traits (ARGs, MGEs, and plasmids) by combining:

Antibiotic class risk Clinical relevance Multi-drug resistance (MDR) Likelihood estimates

The final output is a ranked risk categorization for each trait.

Inputs: The script requires the following input files:

1. ARG Data - card_db_raw.csv Contains ARGs and their associated drug classes Required columns: ARG DrugClass (semicolon-separated list of classes)

2. Antibiotic data - antibiotic_class_usage.csv Defines risk scores per antibiotic class Required columns: DrugClass ClassRiskScore

3. Clinically Relevant antibiotics - reference_clinically_relevant_args.csv List of clinically important ARGs Required columns: ARG

4. Mobile Genetic Elements (MGEs) - mges.csv Contains MGE identifiers embedding ARG names (underscore _ separated)

5. Plasmid Data - plasmid_samples.txt Tab-separated file Required columns: Attribute (ARG names) Value (Plasmid ID)

6. Likelihood Data - likelihood_input.csv Provides likelihood scores for traits Required columns: AMR Genomic Trait Number of isolates across source types columns Number of serovars column

Output: Final Output File - Final_RiskScores.csv

Columns Trait - ARG, MGE, or plasmid identifier Risk Severity - Combined ARG/MDR-based score Risk Likelihood - Likelihood value Risk Score Final computed score Risk Category - Rank classification

Dependencies: Python 3.x pandas

The script takes less than 1 minute to run.

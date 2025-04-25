# Creation Of a Large Language Model-Based Review Tool for Follow-Up Of Chest X-Rays

## Project Summary  
This project implements a system for automating the grading of chest X-ray reports using Large Language Models (LLMs). The goal is to classify reports into clinically relevant categories based on the urgency of follow-up required, ranging from normal (grade 1) to critical (grade 5). The system aims to streamline radiology workflow by automating report categorization while preserving the advantages of human oversight.

Two complementary approaches were developed and evaluated:

1. **Semi-Algorithmic Approach**: Extracts structured findings from reports using an LLM, then applies rule-based grading based on finding types, temporal changes (new/stable/worsening), uncertainty levels, and medical device positioning.  

2. **All-LLM Approach**: Leverages an LLM to directly grade reports based on provided guidelines without the intermediate structured extraction step.

Additionally, a third **Judge Approach** uses an LLM to evaluate which of the two results is more appropriate for each report, creating an ensemble system.

## Repository Contents  
- scripts_audit/audit_cxr_v2.py: Python script implementing the grading approaches  
- scripts_audit/results_analysis.ipynb: Jupyter notebook containing analysis of the grading results, including statistical evaluations and visualizations  

## Implementation Details
Grading System (R)  
- R1: Normal without any findings  
- R2: Normal variant or minor pathology, does not require follow-up  
- R3: Abnormal, non-urgent follow-up required 
- R4: Abnormal, potentially important finding  
- R5: Critical, urgent follow-up required  
  
Data and Models  
- Dataset: 200 retrospectively obtained, anonymized CXR reports (100 hospital, 100 primary care)
- Ground Truth: Expert radiologist annotation
- Models: OpenAI o4-mini, Llama3.1-8B
- Evaluation Metrics: Exact agreement, within-1 agreement, Cohen's Kappa, correlation coefficients, MAE, RMSE, and clinical undergrading rates

## Results
All three approaches yielded good results, with a Cohen Kappa of up to 0.466 for the Judge approach and a within-1 agreement of up to 98.0% for the Judge approach. If we group the gradings into three categories – 1/2 (within normal limits), 3 (abnormal) and 4/5 (urgent, actionable), the Cohen Kappa goes up to 0.524. The critical undergrading rate (approach rating <4 when it should be 4/5) was the lowest for the semialgorithmic grading at 5/46 cases.

## Usage
To use these scripts:

1. Set up the required environment with dependencies
2. Update API keys in the script (OpenAI key if necessary)
3. Run the analysis notebook to evaluate results

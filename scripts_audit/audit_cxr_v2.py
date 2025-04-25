#%%
# Initial settings & imports
##############################
### LLM CXR GRADING SCRIPT ###
##############################

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re
from llm_iter import getLLMJSON, loopEvaluate
from llama_index.llms.ollama import Ollama
from llama_index.core.bridge.pydantic import BaseModel
import ast

# Consider the following prompt
'''
You are well versed in medical terminology for Chest X Ray reports, and your task is to review the given report and grade the report on a scale of 1 to 6, to 1 decimal point.

You will do the above by

1. listing out findings
2. assigning if each findings requires any follow up and at which interval
3. give the grading by considering all findings

Enlarged heart size only needs follow up if other related changes.

Unfolded aorta, vascular calcifications, chronic bone findings, scarring changes do NOT need any follow up.
Benign findings also do NOT require follow up.
Patient rotation and suboptimal effort do NOT require follow up.

Pleural thickening and atelectasis can be followed up after months, unless otherwise qualified.

Potential infections and possible fluid overload or pulmonary congestion are always actionable within 1 week.
Any changes which have increased from before are actionable within 1 week.

Pneumoperitoneum, aortic dissection or pneumothorax of any size are critical emergencies! P1 remarks are also emergencies!

Grading (R):
1 = normal without any findings
2 = normal variant or minor pathology, does not require follow up
3 = abnormal, non-urgent follow-up required
4 = abnormal, potentially important finding
5 = critical, urgent follow-up required

Mainly use the highest grade of a finding, especially for grades 4 or 5.
You may use an average between grade (such as 2.5, 3.5 etc) for other grades.
'''

# Our task for this script
'''
First, we classify the list of reports into positive findings
- Pathological findings
    > Lung
    > Pleura
    > Heart
    > Mediastinum
    > MSK
    > Abdomen
- Tube and line findings (malpositioning)

Then whether the finding is new, better, worse or stable

Below is a sample of an X-ray report, enclosed in three backticks (```). Please generate a list of positive findings from the report, and whether the finding is new, better, worse or stable.
'''

'''
Notes:
- If the chronicity of a finding is not mentioned, follow the path of "new"
- If a finding is "probably" X, it should be considered as X
- Find any P1 and P2SMS findings
- Find either new or worsening findings - should be R5
- Better or stable findings should be downgraded by one grade? R3?
- Uncertain findings can be downgraded by one grade
- If the only finding is "heart size cannot be assessed", it should be R2
- If the only finding is suboptimal technique, it should be R2
- Unfolded aorta should be R2
    - Is this the only situation for R2?
- Any line present should be at least R3 (unless malpositioned)
    - NGT can be R3
    - Any other line should be at least R4 (because they're seen in critically ill pts)
    - Malpositioned lines are all R5 (P1)
'''

#llm = Ollama(model="llama3.1:latest", request_timeout=120, temperature=0.2)
llm = OpenAI(model="o4-mini", temperature=0.2, api_key=os.getenv("GPT_KEY"))

sys_role_msg = 'You are a helpful medical assistant. Please provide your answer in a valid JSON.'
sys_role_chat_msg = ChatMessage(role="system", content=sys_role_msg)

sample_report = '''XR, Chest, PA/AP of 26-JUN-2022:

Patchy airspace changes in the right lung could be in keeping with aspiration pneumonia. Airspace changes in the left lower zone have improved. The heart size is not accurately assessed. Tracheostomy tube and feeding tube seen in situ.'''

with open("padchest_2.json", "r") as f:
    padchest = json.load(f)
    
with open("padchest_tubes_lines.json", "r") as f:
    tubes_lines = json.load(f)

with open("diagnoses.json", "r") as f:
    diagnoses = json.load(f)

df_reports = pd.read_csv("../data_audit/200_sample.csv")
df_reports['R'] = df_reports['R'].map(lambda x: 3 if x in [3, 4] else 4 if x == 5 else 5 if x == 6 else x)


#%%
# Method 1 - Semi-algorithmic
##################################
### METHOD 1: SEMI-ALGORITHMIC ###
##################################

padchest_findings = padchest.keys()
padchest_findings_str = ", ".join(padchest_findings)

tubes_lines_findings = tubes_lines.keys()
tubes_lines_findings_str = ", ".join(tubes_lines_findings)

diagnoses_findings = diagnoses.keys()
diagnoses_str = ", ".join(diagnoses_findings)

def format_prompt_findings(sample_report):
    
    prompt_findings = f'''# Your Task
    Below is a sample of an X-ray report, enclosed in <report> tags. 

    <report>
    {sample_report}
    </report>

    Please generate a list of positive findings from the report, and whether the finding is new, better, worse or stable. 
    Also, please specify if any medical devices (tubes & lines) are present in the report and their placement, whether it is satisfactory or malpositioned.
    Medical devices should not be in "findings" but in "devices". If there is no medical device, please return an empty list for "devices_all" ([]).
    Mastectomy, surgical clips and surgical plates should be considered as findings and not as medical devices.
    
    Please choose the finding from this list. \
    Use only the term that best fits the finding. \
    Do not return a finding that is not in this list. \
    List all positive findings only. Negative findings should not be included.

    Here is the list of findings: {padchest_findings_str}

    Here is a list of medical devices (tubes & lines): {tubes_lines_findings_str}
    
    Here is a list of overarching diagnoses: {diagnoses_str}

    Here is a description of the parameters:
    For the anatomical findings
    - finding: Pathological finding mentioned in the report, using the list above. Do not return a finding that is not in this list. If you think the finding is not in this list, return the closest one from the list.
    - location: Location of the finding, e.g. right lower zone
    - system: System of the body where the finding is located -- Lung, Pleura, Heart, Mediastinum, MSK, Abdomen
    - temporal: new, better, worse, stable or not mentioned
    - uncertainty: certain, uncertain, not mentioned

    For the medical devices (tubes & lines)
    - medical_device: Name of the tube/line, using the list above. Do not return a device that is not in this list.
    - placement: Placement of the tube/line -- satisfactory, suboptimal (abnormal but no urgent action is required), malpositioned (urgent repositioning or removal is required) or not mentioned

    For the diagnoses
    - diagnosis: Overarching diagnosis suggested in the report, using the list above. Do not return a diagnosis that is not in this list.
        - For example, if "...suggestive of infection" is mentioned, return "pneumonia" as the diagnosis.
        - For example, if "...may be related to infection" is mentioned, return "pneumonia" as the diagnosis.
        - For example, if "...malignancy cannot be excluded" is mentioned, return "malignancy" as the diagnosis.
        - For example, if "...P1 for..." is mentioned, return "P1" as the diagnosis.
    - temporal: new, better, worse, stable or not mentioned

    # Finer Points
    If the heart size is not accurately assessed, this should NOT be considered as a finding.
    If the only finding is that there is no consolidation or pleural effusion with no other findings, this should be considered as a normal report with no finding. \
    Even if "no consolidation" or "no pleural effusion" is qualified, for example "no confluent consolidation", \
    "no large pleural effusion" or "no frank pleural effusion" \
    this should be considered as a normal report with no finding.
    If a finding is "probably", "suggestive of", "likely" or any other similar phrase that indicates a low uncertainty, \
    this should be considered as a finding with "certain" uncertainty.
    If a finding is "possibly", "may represent", "could be", "cannot be excluded" \
    or any other similar phrase that indicates a high uncertainty, this should be considered as a finding with "uncertain" uncertainty.
    A rotated or suboptimally inspired film can be considered as "suboptimal study".
    Findings written in pleural should be considered as singular (e.g. "pleural effusions" as "pleural effusion", "nodular opacities" as "nodular opacity", "granulomas" as "granuloma").
    
    If the report suggests the possibility of a diagnosis of pneumonia, infection, or suggests correlation with infective markers, this diagnosis should be raised.
    If the report suggests the possibility of a diagnosis of tuberculosis, atypical infection or mycobacterial infection, this diagnosis should be raised.
    If the report suggests the possibility of a tumour, malignancy, or neoplasm, this diagnosis should be raised.
        - If a CT thorax was suggested for an opacity, the diagnosis is "malignancy".
    If the reporting radiologist indicated at the end that the report is "P1", this diagnosis should be raised.
        - This can come in the format "P1 for X", or "Dr xx was informed at the time of reporting".
        
    If the report suggests that a diagnosis "cannot be excluded", it should be considered as a positive diagnosis.
    If the report suggests that a diagnosis from the diagnoses list is "possible", it should be considered as a positive diagnosis.
    '''
    
    # Extra stuff not needed
    '''
    Airspace opacities, patchy opacities and hazy opacities all count as consolidation. However, perihilar airspace opacities should instead count as edema. 
    Upper lobe diversion and Kerley B lines all count as edema, however Kerley A lines do not. 
    "Prominent pulmonary vasculature" counts as edema.
    Pneumonia counts as consolidation.
    Pulmonary edema, oedema or congestion all count as edema. If "underlying infection cannot be excluded" in the context of opacities, this counts as edema and does not count as consolidation. 
    A nodular opacity or density counts as a nodule. 
    The phrases "collapse/consolidation", "collapse-consolidation" and "collapse/consolidation" should count as collapse and not count as consolidation
    A hydropneumothorax counts as both a pleural effusion and pneumothorax.
    Costophrenic angle blunting should count as a pleural effusion, unless it is mentioned that this may be related to pleural thickening, in which case it should not count as a pleural effusion but count as pleural thickening.
    Fibronodular scarring, fibrocalcific scarring, fibro nodular scarring, and reticulonodular scarring should all count as fibronodular and not count as scarring.  A nodule with adjacent or associated scarring should count as a nodule and scarring, not fibronodular.
    The Chilaiditi sign may be misspelled, for example as "Chiliaditi sign". 
    If a finding "is suggestive", "may represent" or "cannot be excluded", these all count as having the diagnosis.
    Pulmonary fibrosis usually manifests on radiographs as reticular opacities, honeycombing and traction bronchiectasis. It is usually of a specific type, for example "usual interstitial pneumonia" or "nonspecific interstitial pneumonia".
    '''

    return prompt_findings


class PositiveFindings(BaseModel):
    class OneFinding(BaseModel):
        finding: str
        location: str
        system: str
        temporal: str
        uncertainty: str
    
    class Devices(BaseModel):
        medical_device: str
        placement: str
    
    class Diagnoses(BaseModel):
        diagnosis: str
        temporal: str

    findings_all: list[OneFinding]
    devices_all: list[Devices]
    diagnoses_all: list[Diagnoses]

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def closestFinding(txt):
    """
    Given a text, find the closest finding from the padchest findings list.
    """
    # Find the closest finding in padchest_findings
    closest_finding = None
    min_distance = float('inf')
    
    for finding in padchest_findings:
        distance = levenshtein_distance(txt, finding)
        if distance < min_distance:
            min_distance = distance
            closest_finding = finding
    
    return closest_finding

def encode_findings(findings):
    # Make a deep copy to avoid modifying the original
    result = findings.copy()
    temporal_list = ['new', 'better', 'worse', 'stable', 'not mentioned']
    uncertainty_list = ['certain', 'uncertain', 'not mentioned']
    device_placement_list = ['satisfactory', 'suboptimal', 'malpositioned', 'not mentioned']
    
    # Convert each finding to its index in padchest_findings
    if 'findings_all' in result:
        findings_list = list(padchest_findings)  # Convert dict keys to list
        for one_finding in result['findings_all']:
            if one_finding['finding'] in findings_list:
                one_finding['finding'] = findings_list.index(one_finding['finding'])
            else:
                # If the finding is not in the list, find the closest one
                closest_finding = closestFinding(one_finding['finding'])
                if closest_finding:
                    one_finding['finding'] = findings_list.index(closest_finding)

            if one_finding['temporal'] in temporal_list:
                one_finding['temporal'] = temporal_list.index(one_finding['temporal'])
            if one_finding['uncertainty'] in uncertainty_list:
                one_finding['uncertainty'] = uncertainty_list.index(one_finding['uncertainty'])
    
    # Convert each device to its index in tubes_lines_findings
    if 'devices_all' in result:
        devices_list = list(tubes_lines_findings)  # Convert dict keys to list
        for one_device in result['devices_all']:
            if one_device['medical_device'] in devices_list:
                one_device['medical_device'] = devices_list.index(one_device['medical_device'])
            if one_device['placement'] in device_placement_list:
                one_device['placement'] = device_placement_list.index(one_device['placement'])

    # Convert each diagnosis to its index in diagnoses
    if 'diagnoses_all' in result:
        diagnoses_list = list(diagnoses)  # Convert dict keys to list
        for one_diagnosis in result['diagnoses_all']:
            if one_diagnosis['diagnosis'] in diagnoses_list:
                one_diagnosis['diagnosis'] = diagnoses_list.index(one_diagnosis['diagnosis'])
            if one_diagnosis['temporal'] in temporal_list:
                one_diagnosis['temporal'] = temporal_list.index(one_diagnosis['temporal'])
    
    return result

def semanticExtractionCXR(rpt):
    prompt_findings_chat_msg = ChatMessage(role="user", content=format_prompt_findings(rpt))
    messages_lst = [sys_role_chat_msg, prompt_findings_chat_msg]
    findings_devices = getLLMJSON(messages_lst, llm, PositiveFindings)
    findings_devices_encoded = encode_findings(findings_devices)

    return findings_devices_encoded

#print(semanticExtractionCXR(sample_report))

def get_findings_and_lines(row):
    result = semanticExtractionCXR(row['REPORT'])
    
    # Debug logging
    print(f'Report: {row["REPORT"]}\n')
    print('Findings:', end=' ')
    if 'findings_all' in result and result['findings_all']:
        findings_list = []
        for finding in result['findings_all']:
            one_finding = finding['finding']
            if isinstance(one_finding, int):
                finding_text = list(padchest_findings)[one_finding]
                findings_list.append(f"({one_finding}) {finding_text}")
            else:
                findings_list.append(f'!{one_finding}!')
        print(', '.join(findings_list))
    else:
        print("No findings detected")
        
    print('Devices: ', end=' ')
    if 'devices_all' in result and result['devices_all']:
        devices_list = []
        for device in result['devices_all']:
            one_device = device['medical_device']
            if isinstance(one_device, int):
                device_text = list(tubes_lines_findings)[one_device]
                devices_list.append(f"({one_device}) {device_text}")
            else:
                devices_list.append(f'!{one_device}!')
        print(', '.join(devices_list))
    else:
        print("No devices detected")
        
    print('Diagnoses:', end=' ')
    if 'diagnoses_all' in result and result['diagnoses_all']:
        diagnoses_list = []
        for diagnosis in result['diagnoses_all']:
            one_diagnosis = diagnosis['diagnosis']
            if isinstance(one_diagnosis, int):
                diagnosis_text = list(diagnoses)[one_diagnosis]
                diagnoses_list.append(f"({one_diagnosis}) {diagnosis_text}")
            else:
                diagnoses_list.append(f'!{one_diagnosis}!')
        print(', '.join(diagnoses_list))
    else:
        print("No diagnoses detected")
    
    print('---')
    
    return pd.Series({
        'report_findings': result['findings_all'] if 'findings_all' in result else None,
        'report_lines': result['devices_all'] if 'devices_all' in result else None,
        'report_diagnoses': result['diagnoses_all'] if 'diagnoses_all' in result else None,
    })

# Function to safely parse a string representation of a list of dictionaries
def parse_list_dict(val):
    # If already a list or NumPy array, return it as a list.
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    
    # If the value is not a string, return empty list
    if not isinstance(val, str):
        return []
    
    # Check for common "null" values
    if val in (None, 'None'):
        return []
    
    # Check for a missing value using pd.isna
    if pd.isna(val):
        return []
    
    # Try to parse using ast.literal_eval
    try:
        return ast.literal_eval(val)
    except Exception:
        try:
            cleaned_str = val.replace("'", '"').replace("None", "null")
            return json.loads(cleaned_str)
        except Exception:
            print(f"Failed to parse: {val}")
            return []

# Function to extract priorities from findings and lines
def get_priorities(row):
    findings_priorities = []
    lines_priorities = []
    diagnoses_priorities = []
    
    # Parse findings
    findings = parse_list_dict(row['report_findings'])
    if findings:
        for finding in findings:
            finding_index = finding.get('finding')
            if isinstance(finding_index, int):
                finding_name = list(padchest.keys())[finding_index]
                priority = padchest[finding_name]
                # Check if temporal is 'new' (0) or 'worse' (2) and increase priority by 1 if so
                temporal = finding.get('temporal')
                if temporal in [0, 2]:  # 0 is 'new', 2 is 'worse'
                    priority += 1
                if temporal in [1, 3]:  # 1 is 'better', 3 is 'stable'
                    priority -= 1

                # Check if uncertainty is 'uncertain' (1) and decrease priority by 1
                uncertainty = finding.get('uncertainty')
                if uncertainty == 1:  # 1 is 'uncertain'
                    priority -= 1

                # Ensure priority is within the range of 1 to 5
                priority = max(1, min(priority, 5))

                # Append the priority to the list
                findings_priorities.append(priority)
    
    # Parse lines/devices
    lines = parse_list_dict(row['report_lines'])
    if lines:
        for line in lines:
            device_index = line.get('medical_device')
            if isinstance(device_index, int):
                device_name = list(tubes_lines.keys())[device_index]
                priority = tubes_lines[device_name]
                # Check if placement is 'malpositioned' (2) and change priority to 5
                placement = line.get('placement')
                if placement == 2:
                    priority = 5
                
                # Append the priority to the list
                lines_priorities.append(priority)
    
    # Parse diagnoses
    all_dx = parse_list_dict(row['report_diagnoses'])
    if all_dx:
        for diagnosis in all_dx:
            diagnosis_index = diagnosis.get('diagnosis')
            if isinstance(diagnosis_index, int):
                diagnosis_name = list(diagnoses.keys())[diagnosis_index]
                priority = diagnoses[diagnosis_name]

                # Check if temporal is 'better' (1) or 'stable' (3) and increase priority by 1 if so
                temporal = diagnosis.get('temporal')
                if temporal in [1, 3]:  # 1 is 'better', 3 is 'stable'
                    priority -= 1
                
                # Ensure priority is within the range of 1 to 5
                priority = max(1, min(priority, 5))

                # Append the priority to the list
                diagnoses_priorities.append(priority)
    
    # Create a dictionary to hold the new values
    new_values = {
        'findings_priorities': findings_priorities if findings_priorities else [1],
        'lines_priorities': lines_priorities if lines_priorities else [1],
        'diagnoses_priorities': diagnoses_priorities if diagnoses_priorities else [1],
    }
    
    # Add max priorities if applicable
    # Set max_finding_priority to max of findings_priorities or 1 (normal) if empty (i.e. no findings)
    if findings_priorities:
        new_values['max_finding_priority'] = max(findings_priorities)
    else:
        new_values['max_finding_priority'] = 1
    
    # Do the same for the lines findings
    if lines_priorities:
        new_values['max_line_priority'] = max(lines_priorities)
    else:
        new_values['max_line_priority'] = 1
        
    
    if diagnoses_priorities:
        new_values['max_diagnosis_priority'] = max(diagnoses_priorities)
    else:
        new_values['max_diagnosis_priority'] = 1

    # Use findings_priorities and lines_priorities to set overall_max_priority
    # If both are empty (should not happen), set overall_max_priority to 1 
    if findings_priorities or lines_priorities:
        new_values['overall_max_priority'] = max(findings_priorities + lines_priorities + diagnoses_priorities)
    else:
        new_values['overall_max_priority'] = 1
    
    if not findings_priorities and not lines_priorities:
        new_values['overall_max_priority'] = 1
    
    return pd.Series(new_values)
# %%
# Semiautomatic grading
df_reports_semiauto = df_reports.copy()
df_reports_semiauto[['report_findings', 'report_lines', 'report_diagnoses']] = df_reports_semiauto.apply(get_findings_and_lines, axis=1)
df_reports_semiauto.to_csv("../data_audit/200_semiauto.csv", index=False)

# Apply the function to each row
new_columns = df_reports_semiauto.apply(get_priorities, axis=1)
df_reports_semiauto = pd.concat([df_reports_semiauto, new_columns], axis=1)
# Save the updated DataFrame to a new CSV file
df_reports_semiauto.to_csv("../data_audit/200_semiauto_classified.csv", index=False)

# %%
# LLM Grading
#############################
### METHOD 2: LLM GRADING ###
#############################

prompt_grade_cxr = '''Below is a chest X-ray report, enclosed in three backticks (```). \
Please grade the report on a scale of 1 to 6, to 1 decimal point.
Return the highest grade of the findings in the report.

Grading (R):
1 = normal without any findings
2 = normal variant or minor pathology, does not require follow up
3 = abnormal, non-urgent follow-up required
4 = abnormal, potentially important finding
5 = critical, urgent follow-up required

Grade 5 should be used for critical, life threatening findings, including:
- Pneumothorax
- Aortic dissection
- Mediastinal emphysema
- Pneumoperitoneum
- Portal venous air
- Dilated bowel loops
- Significant tube or line misplacement e.g. malpositioned NGT

Grade 5 should also be used unexpected, life-changing findings, including:
- New lesions that are suspicious for malignancy
- New fracture
- New large volume pleural effusions
- New aneurysm

Any finding that has significant clinical impact that is new or worsening should be graded as 4 \
(save for those to be graded as 2, as below).
If the chronicity of a finding is not mentioned, it should be considered as new.
If a finding is "probably" X, it should be considered as X.

Enlarged heart size only needs follow up if other related changes.

Unfolded aorta, vascular calcifications, chronic bone findings, lung scarring and an unfolded thoracic aorta do NOT need any follow up and should be graded as 2.
Benign findings also do NOT require follow up and should be graded as 2.
Patient rotation and suboptimal effort do NOT require follow up and should be graded as 2.
If there is suboptimal technique with no other finding, this should be graded as 2.

Pleural thickening and atelectasis should be grade 3, unless otherwise qualified.

Potential infection/consolidation and possible fluid overload or pulmonary congestion should be graded as 4.
Patchy opacities, airspace opacities and hazy opacities should be graded as 4 for new pneumonia/infection.
A small pleural effusion can be graded as 3, but significant ones (moderate/severe) should be graded as 4.
If the report suggests the possibility of a diagnosis of pneumonia, infection, or suggests correlation with infective markers, this should be graded as 4.
If the report suggests the possibility of a diagnosis of tuberculosis, atypical infection or mycobacterial infection, this should be graded as 4.
    - If the report suggests that tuberculosis is stable, this should be graded as 3.
If the report suggests the possibility of a tumour, malignancy, or neoplasm, this should be graded as 4.
    - If the report suggests that the tumour is stable, this should be graded as 3.
If the reporting radiologist indicated at the end that the report is "P1", this should be graded as 5.
    - This can come in the format "P1 for X", or "Dr xx was informed at the time of reporting".

Any line present should be at least grade 3 (unless malpositioned, in which case it should be grade 6)
    - A nasogastric tube in a satisfactory position can be grade 3
    - Any other line should be at least grade 4 (because they're seen in critically ill pts)
    - Malpositioned lines are all grade 5.

Please provide your answer in a valid JSON with the following format:
{
    "grade": float (1 to 5, to 1 decimal point)
}

Please grade this CXR report:
'''

class cxrGrade(BaseModel):
    grade: float

def gradeCXR(rpt):
    prompt_grade_chat_msg = ChatMessage(role="user", content=f'{prompt_grade_cxr}\n\n```\n{rpt}\n```')
    messages_lst = [sys_role_chat_msg, prompt_grade_chat_msg]

    return getLLMJSON(messages_lst, llm, cxrGrade)

df_reports['report_grade'] = df_reports.apply(lambda row: gradeCXR(row['Report_Text'])['grade'], axis=1)

df_reports.to_csv("../data_audit/200_llm.csv", index=False)
# %%
# Merge everything and save the final CSV
############################
### MERGE ALL DATAFRAMES ###
############################

df_llm_graded = pd.read_csv("../data_audit/200_sample_regraded.csv")
# Keep only the accession column and overall_max_priority
df_llm_graded = df_llm_graded[['Accession_No', 'report_grade']].copy()
# Convert the report_grade column to integer
df_llm_graded['report_grade'] = pd.to_numeric(df_llm_graded['report_grade'], errors='coerce').astype(int)
# Rename report_grade to priority_llm
df_llm_graded = df_llm_graded.rename(columns={'report_grade': 'priority_llm'})

df_algo_graded = pd.read_csv("../data_audit/200_semiauto_classified.csv")
# Keep only the  accession column and overall_max_priority
df_algo_graded = df_algo_graded[['Accession_No', 'overall_max_priority']].copy()
# Rename overall_max_priority to priority_algo
df_algo_graded = df_algo_graded.rename(columns={'overall_max_priority': 'priority_algo'})
# Convert the priority_algo column to numeric
df_algo_graded['priority_algo'] = pd.to_numeric(df_algo_graded['priority_algo'], errors='coerce')

df_reports_overall = pd.read_csv("../data_audit/200_sample_regraded.csv")
# Rename the R column to priority_manual
df_reports_overall = df_reports_overall.rename(columns={'R': 'priority_manual'})
# Convert the priority column to numeric
df_reports_overall['priority_manual'] = pd.to_numeric(df_reports_overall['priority_manual'], errors='coerce')

# Merge all three dataframes on Accession_No
df_reports_overall = pd.merge(df_reports_overall, df_llm_graded, on='Accession_No', how='left')
df_reports_overall = pd.merge(df_reports_overall, df_algo_graded, on='Accession_No', how='left')

# Save the merged dataframe to a CSV file
df_reports_overall.to_csv("../data_audit/200_overall.csv", index=False)
# %%
# We try LLM as judge
###################################
### TRIAL: LLM AS JUDGE GRADING ###
###################################

#df_reports_overall = pd.read_csv("../data_audit/100_findings_all_v2.csv")

llm_reasoning = OpenAI(model="o4-mini", temperature=0.2, api_key=os.getenv("GPT_KEY"), additional_kwargs={"reasoning_effort": "high"})

comparison_gudelines = '''Grading (R):
    1 = normal without any findings
    2 = normal variant or minor pathology, does not require follow up
    3 = abnormal, non-urgent follow-up required
    4 = abnormal, potentially important finding
    5 = critical, urgent follow-up required

    Grade 5 should be used for critical, life threatening findings, including:
    - Pneumothorax
    - Aortic dissection
    - Mediastinal emphysema
    - Pneumoperitoneum
    - Portal venous air
    - Dilated bowel loops
    - Significant tube or line misplacement e.g. malpositioned NGT

    Grade 5 should also be used unexpected, life-changing findings, including:
    - New lesions that are suspicious for malignancy
    - New fracture
    - New large volume pleural effusions
    - New aneurysm
'''

def formatPromptComparison(rpt, grade_algo, grade_llm):
    prompt_judge_grading = f'''Your task is to grade the following CXR report, enclosed in three backticks (```). \
    You are given two automated grades (algorithm and LLM). \
    Please tell me which of the two automated grades is more appropriate for this report, and explain your answer. 
    Please grade the report on a scale of 1 to 6.

    Below are some guidelines:
    {comparison_gudelines}

    Please provide your answer in a valid JSON with the following format:
    {{
        "grade": 1-5, what you think is the most appropriate grade,
        "choice": 0-3 - 
            0 = both are not appropriate
            1 = algorithm grade is better
            2 = LLM grade is better
            3 = both grades are the same
        "explanation": your explanation of your responses above
    }}

    Please grade this CXR report:
    {rpt}

    The algorithm grade is {grade_algo} and the LLM grade is {grade_llm}.
    '''

    return prompt_judge_grading

def formatPromptComparison_GT(rpt, grade_auto, grade_manual):
    prompt_judge_grading = f'''Your task is to grade the following CXR report, enclosed in three backticks (```). \
    You are given two grades (algorithm and manual). \
    Please tell me which of the two grades is more appropriate for this report, and explain your answer. 
    Please grade the report on a scale of 1 to 5.

    Below are some guidelines:
    {comparison_gudelines}

    Please provide your answer in a valid JSON with the following format:
    {{
        "grade": 1-5, what you think is the most appropriate grade,
        "choice": 0-2 - 
            0 = both are not appropriate
            1 = algorithm grade is better
            2 = manual grade is better
            3 = both grades are the same
        "explanation": your explanation of your responses above
    }}

    Please grade this CXR report:
    {rpt}

    The algorithm grade is {grade_auto} and the manual grade is {grade_manual}.
    '''

    return prompt_judge_grading

class cxrGradeJudge(BaseModel):
    grade: int
    choice: int
    explanation: str

def judgeGrading(row):
    rpt = row['REPORT']
    grade_algo = row['priority_algo']
    grade_llm = row['priority_llm']

    return_json = {}

    # If the algo and LLM grades are the same, return the grade and choice 3
    if grade_algo == grade_llm:
        return_json['grade_int'] = grade_algo
        return_json['choice_int'] = 3
        return_json['explanation_int'] = None
    # Otherwise, we use the LLM to judge which is better
    else:
        prompt_judge_chat_msg = ChatMessage(role="user", content=formatPromptComparison(rpt, grade_algo, grade_llm))
        messages_lst = [sys_role_chat_msg, prompt_judge_chat_msg]
        grade_json = getLLMJSON(messages_lst, llm_reasoning, cxrGradeJudge)
        return_json['grade_int'] = grade_json['grade']
        return_json['choice_int'] = grade_json['choice']
        return_json['explanation_int'] = grade_json['explanation']
    
    # If the algo and manual grades are the same, return the grade and choice 3
    if row['priority_manual'] == return_json['grade_int']:
        return_json['grade_ext'] = return_json['grade_int']
        return_json['choice_ext'] = 3
        return_json['explanation_ext'] = None
    # Otherwise, we use the LLM to judge which is better
    else:
        prompt_judge_chat_msg = ChatMessage(role="user", content=formatPromptComparison_GT(rpt, grade_algo, row['priority_manual']))
        messages_lst = [sys_role_chat_msg, prompt_judge_chat_msg]
        grade_json = getLLMJSON(messages_lst, llm_reasoning, cxrGradeJudge)
        return_json['grade_ext'] = grade_json['grade']
        return_json['choice_ext'] = grade_json['choice']
        return_json['explanation_ext'] = grade_json['explanation']
    
    return return_json

# Apply judgeGrading function to each row and extract results
def add_judge_columns(row):
    result = judgeGrading(row)
    print(f'Report: {row["REPORT"]}')
    print(f'LLM/Algo (GT) grading: {row["priority_llm"]}/{row["priority_algo"]} ({row["priority_manual"]})')
    print(f'Judge grading: {result["grade_int"]}\n')
    print(f'Judge reasoning: {result["explanation_int"]}\n')
    # Print which grade the LLM chose (manual or auto) based on choice_ext
    if result['choice_ext'] == 0:
        print("LLM chose: Neither grade is appropriate")
    elif result['choice_ext'] == 1:
        print("LLM chose: Algorithm grade is better")
    elif result['choice_ext'] == 2:
        print("LLM chose: Manual grade is better")
    elif result['choice_ext'] == 3:
        print("LLM chose: Both grades are the same")
    print('---')
    return pd.Series({
        'judge_grade': result['grade_int'],
        'judge_choice': result['choice_int'],
        'judge_reasoning': result['explanation_int'],
        'judge_grade_ext': result['grade_ext'],
        'judge_choice_ext': result['choice_ext'],
        'judge_reasoning_ext': result['explanation_ext'],  
    })

# Add the new columns to df_reports_overall
judge_columns = df_reports_overall.apply(add_judge_columns, axis=1)
df_reports_overall = pd.concat([df_reports_overall, judge_columns], axis=1)

# Save the updated dataframe
df_reports_overall.to_csv("../data_audit/200_judged.csv", index=False)

from lib_audit_cxr_v2 import CXRClassifier
import pandas as pd
import numpy as np
import os
import re
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm

def clean_text(text):
    """
    Cleans the input text by removing unwanted characters and formatting.
    """
    # Remove the header
    text = re.sub(r'CHEST\sof\s\d+-\S{3}-\d{4}:', '', text)
    text = re.sub(r'Accession\sNo:\S+', '', text)
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Remove leading and trailing spaces
    text = text.strip()
    return text

def extract_grade(result):
    """Helper function to safely extract grade from result"""
    if isinstance(result, dict):
        return result.get('grade', 0)
    return 0

def extract_hybrid_result(result):
    """Helper function to safely extract hybrid grading result"""
    if isinstance(result, dict):
        return result.get('grade', 0), result.get('explanation', '')
    return 0, ''

class BatchCXRProcessor:
    """
    Concurrent batch processor for CXR reports using CXRClassifier.
    
    This class provides methods to process multiple CXR reports concurrently,
    significantly speeding up the I/O limited LLM processing steps.
    """
    
    def __init__(self, 
                 findings_dict: Dict, 
                 tubes_lines_dict: Dict, 
                 diagnoses_dict: Dict,
                 model_name: str = "gpt-4o-mini",
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_workers: int = 5,
                 rate_limit_delay: float = 0.1):
        """
        Initialize the batch processor.
        
        Args:
            findings_dict: Dictionary of medical findings
            tubes_lines_dict: Dictionary of tubes and lines
            diagnoses_dict: Dictionary of diagnoses
            model_name: Name of the model to use
            base_url: Base URL for the API (for Ollama use: "http://localhost:11434/v1")
            api_key: API key (use "dummy" for Ollama)
            max_workers: Maximum number of concurrent workers
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        self.findings_dict = findings_dict
        self.tubes_lines_dict = tubes_lines_dict
        self.diagnoses_dict = diagnoses_dict
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        
        # Create a classifier instance for each thread
        self._classifiers = {}
        self._lock = threading.Lock()
    
    def _get_classifier(self) -> CXRClassifier:
        """Get a thread-local CXRClassifier instance."""
        thread_id = threading.get_ident()
        
        if thread_id not in self._classifiers:
            with self._lock:
                if thread_id not in self._classifiers:
                    self._classifiers[thread_id] = CXRClassifier(
                        findings=self.findings_dict,
                        tubes_lines=self.tubes_lines_dict,
                        diagnoses=self.diagnoses_dict,
                        model_name=self.model_name,
                        base_url=self.base_url,
                        api_key=self.api_key
                    )
        
        return self._classifiers[thread_id]
    
    def _process_single_report_semialgo(self, report_text: str, index: int) -> Tuple[int, Dict]:
        """Process a single report using semi-algorithmic method."""
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            classifier = self._get_classifier()
            result = classifier.gradeReportSemialgo(report_text)
            return index, result
        except Exception as e:
            print(f"Error processing report {index} (semialgo): {e}")
            return index, {}
    
    def _process_single_report_llm(self, report_text: str, index: int) -> Tuple[int, Dict]:
        """Process a single report using LLM method."""
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            classifier = self._get_classifier()
            result = classifier.gradeReportLLM(report_text)
            return index, result
        except Exception as e:
            print(f"Error processing report {index} (LLM): {e}")
            return index, {}
    
    def _process_single_report_hybrid(self, report_text: str, grade_semialgo: int, index: int) -> Tuple[int, Dict]:
        """Process a single report using hybrid method."""
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            classifier = self._get_classifier()
            result = classifier.gradeReportHybrid(report_text, grade_semialgo)
            return index, result
        except Exception as e:
            print(f"Error processing report {index} (hybrid): {e}")
            return index, {}
    
    def _process_single_report_judge(self, report_text: str, manual_grade: int, algo_grade: int, llm_grade: int, index: int) -> Tuple[int, Dict]:
        """Process a single report using judge method."""
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            classifier = self._get_classifier()

            # Call gradeReportJudge which returns a pandas Series
            result_dict = classifier.gradeReportJudge(report_text, manual_grade, algo_grade, llm_grade)

            return index, result_dict
        except Exception as e:
            print(f"Error processing report {index} (judge): {e}")
            # Return default values matching the expected output
            return index, {
                'judge_grade': 0,
                'judge_choice': 0, 
                'judge_reasoning': '',
                'judge_grade_ext': 0,
                'judge_choice_ext': 0,
                'judge_reasoning_ext': ''
            }
    
    def process_semialgo_batch(self, df: pd.DataFrame, report_column: str = 'REPORT') -> pd.DataFrame:
        """
        Process a batch of reports using semi-algorithmic method concurrently.
        
        Args:
            df: DataFrame containing reports
            report_column: Name of the column containing report text
            
        Returns:
            DataFrame with added semi-algorithmic grading columns
        """
        print(f"Processing {len(df)} reports with semi-algorithmic method using {self.max_workers} workers...")
        
        df_result = df.copy()
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single_report_semialgo, row[report_column], idx): idx 
                for idx, row in df.iterrows()
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Semi-algo processing"):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as e:
                    index = future_to_index[future]
                    print(f"Error in future for index {index}: {e}")
                    results[index] = {}
        
        # Add results to dataframe
        semialgo_data = []
        for idx in df.index:
            if idx in results:
                semialgo_data.append(results[idx])
            else:
                semialgo_data.append({})
        
        semialgo_df = pd.DataFrame(semialgo_data, index=df.index)
        return pd.concat([df_result, semialgo_df], axis=1)
    
    def process_llm_batch(self, df: pd.DataFrame, report_column: str = 'REPORT') -> pd.DataFrame:
        """
        Process a batch of reports using LLM method concurrently.
        
        Args:
            df: DataFrame containing reports
            report_column: Name of the column containing report text
            
        Returns:
            DataFrame with added LLM grading column
        """
        print(f"Processing {len(df)} reports with LLM method using {self.max_workers} workers...")
        
        df_result = df.copy()
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single_report_llm, row[report_column], idx): idx 
                for idx, row in df.iterrows()
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="LLM processing"):
                try:
                    index, result = future.result()
                    results[index] = extract_grade(result)
                except Exception as e:
                    index = future_to_index[future]
                    print(f"Error in future for index {index}: {e}")
                    results[index] = 0
        
        # Add results to dataframe
        df_result['llm_grade'] = [results.get(idx, 0) for idx in df.index]
        return df_result
    
    def process_hybrid_batch(self, df: pd.DataFrame, report_column: str = 'REPORT', 
                           semialgo_column: str = 'overall_max_priority') -> pd.DataFrame:
        """
        Process a batch of reports using hybrid method concurrently.
        
        Args:
            df: DataFrame containing reports and semi-algo grades
            report_column: Name of the column containing report text
            semialgo_column: Name of the column containing semi-algo grades
            
        Returns:
            DataFrame with added hybrid grading columns
        """
        print(f"Processing {len(df)} reports with hybrid method using {self.max_workers} workers...")
        
        df_result = df.copy()
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single_report_hybrid, 
                               row[report_column], row[semialgo_column], idx): idx 
                for idx, row in df.iterrows()
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Hybrid processing"):
                try:
                    index, result = future.result()
                    grade, explanation = extract_hybrid_result(result)
                    results[index] = {'grade': grade, 'explanation': explanation}
                except Exception as e:
                    index = future_to_index[future]
                    print(f"Error in future for index {index}: {e}")
                    results[index] = {'grade': 0, 'explanation': ''}
        
        # Add results to dataframe
        df_result['hybrid_grade'] = [results.get(idx, {}).get('grade', 0) for idx in df.index]
        df_result['hybrid_explanation'] = [results.get(idx, {}).get('explanation', '') for idx in df.index]
        return df_result
    
    def process_judge_batch(self, df: pd.DataFrame, report_column: str = 'REPORT',
                          manual_column: str = 'priority_manual',
                          algo_column: str = 'priority_algo', 
                          llm_column: str = 'priority_llm') -> pd.DataFrame:
        """
        Process a batch of reports using judge method concurrently.
        
        Args:
            df: DataFrame containing reports and existing grades
            report_column: Name of the column containing report text
            manual_column: Name of the column containing manual grades
            algo_column: Name of the column containing algorithmic grades
            llm_column: Name of the column containing LLM grades
            
        Returns:
            DataFrame with added judge grading columns
        """
        print(f"Processing {len(df)} reports with judge method using {self.max_workers} workers...")
        
        df_result = df.copy()
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks using the unified single report method
            future_to_index = {
                executor.submit(self._process_single_report_judge, 
                               row[report_column], row[manual_column], 
                               row[algo_column], row[llm_column], idx): idx 
                for idx, row in df.iterrows()
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Judge processing"):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as e:
                    index = future_to_index[future]
                    print(f"Error in future for index {index}: {e}")
                    results[index] = {
                        'judge_grade': 0,
                        'judge_choice': 0,
                        'judge_reasoning': '',
                        'judge_grade_ext': 0,
                        'judge_choice_ext': 0,
                        'judge_reasoning_ext': ''
                    }
        
        # Add results to dataframe - now using the consistent approach
        for column in ['judge_grade', 'judge_choice', 'judge_reasoning', 
                      'judge_grade_ext', 'judge_choice_ext', 'judge_reasoning_ext']:
            df_result[column] = [results.get(idx, {}).get(column, 0 if 'grade' in column or 'choice' in column else '') 
                                for idx in df.index]
        
        return df_result
    
    def process_full_pipeline(self, df: pd.DataFrame, report_column: str = 'REPORT',
                            save_intermediate: bool = True, output_dir: str = './') -> pd.DataFrame:
        """
        Process the full pipeline: semialgo -> hybrid -> LLM -> judge
        
        Args:
            df: DataFrame containing reports
            report_column: Name of the column containing report text
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save intermediate files
            
        Returns:
            DataFrame with all grading columns added
        """
        print("Starting full CXR processing pipeline...")
        start_time = time.time()
        
        # Clean reports
        df_processed = df.copy()
        df_processed[report_column] = df_processed[report_column].apply(clean_text)
        
        # Step 1: Semi-algorithmic processing
        print("\n=== Step 1: Semi-algorithmic Processing ===")
        df_processed = self.process_semialgo_batch(df_processed, report_column)
        if save_intermediate:
            df_processed.to_csv(os.path.join(output_dir, 'intermediate_semialgo.csv'), index=False)
        
        # Step 2: Hybrid processing
        print("\n=== Step 2: Hybrid Processing ===")
        df_processed = self.process_hybrid_batch(df_processed, report_column)
        if save_intermediate:
            df_processed.to_csv(os.path.join(output_dir, 'intermediate_hybrid.csv'), index=False)
        
        # Step 3: LLM processing
        print("\n=== Step 3: LLM Processing ===")
        df_processed = self.process_llm_batch(df_processed, report_column)
        if save_intermediate:
            df_processed.to_csv(os.path.join(output_dir, 'intermediate_llm.csv'), index=False)
        
        # Step 4: Prepare for judge processing (rename columns if needed)
        if 'priority_manual' not in df_processed.columns and 'GROUND_TRUTH' in df_processed.columns:
            df_processed = df_processed.rename(columns={'GROUND_TRUTH': 'priority_manual'})
        if 'priority_hybrid' not in df_processed.columns and 'hybrid_grade' in df_processed.columns:
            df_processed = df_processed.rename(columns={'hybrid_grade': 'priority_hybrid'})
        if 'priority_algo' not in df_processed.columns and 'overall_max_priority' in df_processed.columns:
            df_processed = df_processed.rename(columns={'overall_max_priority': 'priority_algo'})
        if 'priority_llm' not in df_processed.columns and 'llm_grade' in df_processed.columns:
            df_processed = df_processed.rename(columns={'llm_grade': 'priority_llm'})
        
        # Step 5: Judge processing
        print("\n=== Step 4: Judge Processing ===")
        if all(col in df_processed.columns for col in ['priority_manual', 'priority_algo', 'priority_llm']):
            df_processed = self.process_judge_batch(df_processed, report_column)
        else:
            print("Skipping judge processing - missing required columns")
        
        end_time = time.time()
        print(f"\n=== Pipeline Complete ===")
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        print(f"Average time per report: {(end_time - start_time) / len(df):.2f} seconds")
        
        return df_processed


import os
import pandas as pd
import numpy as np
import time
from facroc import aucc, compute_facroc

def facroc_experiment(dataset_path, clustering_path, figure_out, protected_attr,
                      protected_group, non_protected_group, protected_label, non_protected_label):
    # Load dataset features and clustering labels
    # Assumes dataset_path has columns for features and protected attribute
    # Assumes clustering_path has a 'cluster' column representing cluster assignments
    data = pd.read_csv(dataset_path)
    clustering = pd.read_csv(clustering_path)
    
    # Check if they have the same length
    if len(data) != len(clustering):
        raise ValueError(f"Length mismatch: dataset ({len(data)}) and clustering ({len(clustering)})")
    
    # Align clustering labels with dataset rows (adding 'cluster' column to data for easy slicing)
    data['cluster'] = clustering['cluster'].values
    
    # Slice protected group
    # Can handle numeric encoding (e.g. 0/1) or string groups (e.g. 'Female' / 'Male')
    data_f = data[data[protected_attr] == protected_group].copy()
    
    # Subsample if group size exceeds 3000
    if len(data_f) > 3000:
        data_f = data_f.sample(n=3000, random_state=42)
        
    # Separate features and cluster assignments for protected group
    # Exclude non-feature columns if they exist (e.g., 'cluster', protected_attr, or target variables)
    feature_cols = [col for col in data_f.columns if col not in ['cluster', protected_attr]]
    X_f = data_f[feature_cols].values
    labels_f = data_f['cluster'].values
    
    evaluation_f = aucc(labels_f, X_f, return_rates=True)
    
    # Slice non-protected group
    data_m = data[data[protected_attr] == non_protected_group].copy()
    
    # Subsample if group size exceeds 3000
    if len(data_m) > 3000:
        data_m = data_m.sample(n=3000, random_state=42)
        
    X_m = data_m[feature_cols].values
    labels_m = data_m['cluster'].values
    
    evaluation_m = aucc(labels_m, X_m, return_rates=True)
    
    # Compute FACROC and generate plot
    facroc_score = compute_facroc(
        roc_protected=evaluation_f,
        roc_non_protected=evaluation_m,
        protected_label=protected_label,
        non_protected_label=non_protected_label,
        show_plot=True,
        filename=figure_out
    )
    
    return facroc_score

if __name__ == "__main__":
    # Define datasets matching the project environment
    # Using processed CSV files which are fully numeric and have 'protected_attribute' column
    datasets = [
        {
            "name": "adult",
            "file": "dataset/processed/adult_processed.csv",
            "p_attr": "protected_attribute",
            "p_group": 0, # Female (encoded as 0 in load_adult)
            "np_group": 1, # Male (encoded as 1)
            "p_label": "Female",
            "np_label": "Male"
        },
        {
            "name": "german",
            "file": "dataset/processed/german_processed.csv",
            "p_attr": "protected_attribute",
            "p_group": 0, # Female / non-A91/A93/A94
            "np_group": 1,
            "p_label": "Female",
            "np_label": "Male"
        },
        {
            "name": "compas",
            "file": "dataset/processed/compas_processed.csv",
            "p_attr": "protected_attribute",
            "p_group": 0, # Non-White (Caucasian = 1)
            "np_group": 1, # White
            "p_label": "Non-White",
            "np_label": "White"
        },
        {
            "name": "credit_card",
            "file": "dataset/processed/credit_card_processed.csv",
            "p_attr": "protected_attribute",
            "p_group": 0, # Female (SEX=2 encoded as 0, or SEX=1 as 1)
            "np_group": 1,
            "p_label": "Female",
            "np_label": "Male"
        },
        {
            "name": "student_mat",
            "file": "dataset/processed/student_mat_processed.csv",
            "p_attr": "protected_attribute",
            "p_group": 0, # Female
            "np_group": 1, # Male
            "p_label": "Female",
            "np_label": "Male"
        },
        {
            "name": "student_por",
            "file": "dataset/processed/student_por_processed.csv",
            "p_attr": "protected_attribute",
            "p_group": 0,
            "np_group": 1,
            "p_label": "Female",
            "np_label": "Male"
        }
    ]
    
    methods = ["kmeans", "hierarchical", "fairlet", "scalable_fair", "proportional_fair"]
    
    print("Starting FACROC Experiments...")
    
    # Create output directories for plots and reports
    os.makedirs("reports/figures", exist_ok=True)
    
    for ds in datasets:
        print(f"\nProcessing dataset: {ds['name']}")
        if not os.path.exists(ds['file']):
            print(f"  Missing dataset file: {ds['file']}. Skipping.")
            continue
            
        for method in methods:
            # Assuming clustering results are saved in 'clustering/[method]_[dataset].csv'
            # inside a CSV file containing a column 'cluster'
            clustering_file = f"clustering/{method}_{ds['name']}.csv"
            figure_file = f"reports/figures/{ds['name']}_{method}_facroc.pdf"
            
            if os.path.exists(clustering_file):
                print(f"  Running method: {method}")
                t0 = time.time()
                try:
                    score = facroc_experiment(
                        dataset_path=ds['file'],
                        clustering_path=clustering_file,
                        figure_out=figure_file,
                        protected_attr=ds['p_attr'],
                        protected_group=ds['p_group'],
                        non_protected_group=ds['np_group'],
                        protected_label=ds['p_label'],
                        non_protected_label=ds['np_label']
                    )
                    t1 = time.time()
                    print(f"    FACROC = {score:.4f} (computed in {t1 - t0:.2f}s)")
                except Exception as e:
                    print(f"    Error running {method}: {e}")
            else:
                # To make it easy to test, if no file exists, we print a notice
                # The user can run their clustering models to generate these files first.
                pass

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def _normalize_features(df):
    df = df.copy()
    for col in df.columns:
        # Bỏ qua không chuẩn hóa nếu cột là biến bù (dummy variable) nhị phân chỉ chứa 0 và 1
        unique_vals = df[col].unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            continue
        # Chỉ chuẩn hóa các thuộc tính dạng số liên tục
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        df[col] = (df[col] - mean) / std
    return df

def load_adult():
    data_path = os.path.join(DATASET_DIR, "adult", "adult.data")
    test_path = os.path.join(DATASET_DIR, "adult", "adult.test")
    
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income'
    ]

    df_train = pd.read_csv(data_path, header=None, names=column_names, skipinitialspace=True)
    df_test = pd.read_csv(test_path, header=None, names=column_names, skipinitialspace=True, skiprows=1)
    df_test['income'] = df_test['income'].str.rstrip('.')
    df = pd.concat([df_train, df_test], ignore_index=True)

    df = df.replace('?', np.nan).dropna()
    df = df.drop(columns=['income'])
    P = (df['sex'] == 'Male').astype(int).values
    df = df.drop(columns=['sex'])

    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'native_country']
    df = pd.get_dummies(df, columns=categorical_cols, dtype=float)
    X_df = _normalize_features(df)
    return X_df, P

def load_compas():
    file_path = os.path.join(DATASET_DIR, "compas", "compas-scores-two-years-violent.csv")
    df = pd.read_csv(file_path)

    df = df[
        (df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O')
    ]

    P = (df['race'] == 'Caucasian').astype(int).values

    cols_to_drop = [
        'id', 'name', 'first', 'last', 'compas_screening_date', 'dob',
        'c_jail_in', 'c_jail_out', 'c_offense_date', 'c_arrest_date',
        'r_offense_date', 'r_jail_in', 'r_jail_out', 'vr_offense_date',
        'screening_date', 'v_screening_date', 'in_custody', 'out_custody',
        'start', 'end', 'c_case_number', 'r_case_number', 'vr_case_number',
        'c_charge_desc', 'r_charge_desc', 'vr_charge_desc',
        'r_charge_degree', 'r_days_from_arrest', 'violent_recid', 'vr_charge_degree',
        'two_year_recid', 'two_year_recid.1', 'decile_score.1', 'priors_count.1',
        'score_text', 'v_score_text', 'race'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).dropna()

    categorical_cols = ['sex', 'age_cat', 'c_charge_degree', 'type_of_assessment', 'v_type_of_assessment']
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], dtype=float)
    X_df = _normalize_features(df)
    return X_df, P

def load_credit_card():
    file_path = os.path.join(DATASET_DIR, "default+of+credit+card+clients", "default of credit card clients.xls")
    df = pd.read_excel(file_path, skiprows=1).dropna()

    P = (df['SEX'] == 1).astype(int).values
    df = df.drop(columns=['ID', 'default payment next month', 'SEX'])

    X_df = _normalize_features(df.astype(float))
    return X_df, P

def load_german():
    file_path = os.path.join(DATASET_DIR, "statlog+german+credit+data", "german.data")
    df = pd.read_csv(file_path, sep=r'\s+', header=None).dropna()

    P = df[8].isin(['A91', 'A93', 'A94']).astype(int).values
    df = df.drop(columns=[8, 20])

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, dtype=float)
    X_df = _normalize_features(df)
    return X_df, P

def load_student(subject="mat"):
    filename = f"student-{subject}.csv"
    file_path = os.path.join(DATASET_DIR, "student+performance", filename)
    df = pd.read_csv(file_path, sep=';').dropna()

    P = (df['sex'] == 'M').astype(int).values
    df = df.drop(columns=['sex'])

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, dtype=float)
    X_df = _normalize_features(df)
    return X_df, P
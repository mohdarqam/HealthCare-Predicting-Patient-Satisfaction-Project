import os
import pandas as pd
# import matplotlib.pyplot as plt
import joblib
import pickle
import json
import numpy as np
from django.shortcuts import render



def get_poverty_threshold(family_size):
    base_thresholds = {
        1: 15060,
        2: 20440,
        3: 25820,
        4: 31200
    }
    if family_size in base_thresholds:
        return base_thresholds[family_size]
    else:
        return 31200 + (family_size - 4) * 5380




def dashboard_home(request):
    # Read file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    balanced_path = os.path.join(BASE_DIR, 'dashboard', 'data', 'balanced_df.csv')
    balanced_df = pd.read_csv(balanced_path)
    csv_path = os.path.join(BASE_DIR, 'dashboard', 'data', 'cleaned_df.csv')

    cleaned_df = pd.read_csv(csv_path)

    avg_expenditure = int(cleaned_df['EXPTOT'].mean())

    ins_type_form = "Private Only"  # default type for initial view
    target_insurance_type_enc = lambda x:(
        'Private Only' if x == 0 else
        'Public Only'  if x == 1 else
        'Uninsured' if x == 2 else 3
    )

    filtered_df = cleaned_df[cleaned_df['INSURANCE_TYPE'] == ins_type_form].copy()
    
    insurance_type_encoded = (
    0 if ins_type_form == 'Private Only' else
    1 if ins_type_form == 'Public Only' else
    2 if ins_type_form == 'Uninsured' else 3 )  

    cleaned_filtered_df = balanced_df[balanced_df['INSURANCE_TYPE_ENC'] == insurance_type_encoded].copy()

    if not filtered_df.empty:
        avg_charges_per_head = filtered_df['CHGTOT'].mean()
    else:
        avg_charges_per_head = 0

    # OOP Burden
    avg_oop_burden = round(filtered_df['OOP_BURDEN'].mean() * 100, 2)

    # Catastrophic OOP Rate
    catastrophic_rate = round(filtered_df['CATASTROPHIC_OOP'].mean() * 100, 2)

    # Health label mapping
    health_labels = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
    # avg health score based on insurance type
    avg_health_score = cleaned_filtered_df['HEALTH'].count() 
    # Create dict for all insurance types
    all_health_data = {}
    for ins_type in cleaned_df['INSURANCE_TYPE'].unique():
        df = cleaned_df.loc[cleaned_df['INSURANCE_TYPE'] == ins_type].copy()
        df.loc[:, 'HEALTH_LABEL'] = df['HEALTH'].map(health_labels)
        counts = df['HEALTH_LABEL'].value_counts().reindex(health_labels.values(), fill_value=0).to_dict()
        all_health_data[ins_type] = counts

    # Map short name for display
    if ins_type_form == "Private Only":
        ins_type_display = 'Private'
    elif ins_type_form == "Public Only":
        ins_type_display = 'Public'
    elif ins_type_form == "Mixed":
        ins_type_display = 'Mixed'
    else:
        ins_type_display = 'Uninsured'
    
    avg_health_df = (
    cleaned_filtered_df.groupby('INSURANCE_TYPE_ENC')['HEALTH']
    .mean()
    .reset_index())

    if not avg_health_df.empty:
        match = avg_health_df.loc[
            avg_health_df['INSURANCE_TYPE_ENC'] == insurance_type_encoded, 'HEALTH'
        ]
        avg_health_score = float(match.values[0]) if not match.empty else 0.0
    else:
        avg_health_score = 0.0

    prediction_prob =0
    predicted_health_status= 0
    suitability_status=None



    # Map disease column -> display name
    disease_map = {
    'CANCEREV': 'Cancer',
    'CHOLHIGHEV': 'High Cholesterol',
    'DIABETICEV': 'Diabetes',
    'HEARTCONEV': 'Heart Condition',
    'HYPERTENAGE': 'Hypertension'
    }

    disease_counts = {}
    total_patients = len(df)  # total rows

    for col, name in disease_map.items():
        if col in balanced_df.columns:
            count = int((balanced_df[col] > 0).sum())
            percentage = round((count / total_patients) * 100, 2) if total_patients > 0 else 0
            disease_counts[name] = percentage

    # Convert to lists for chart
    disease_labels = list(disease_counts.keys())
    disease_values = list(disease_counts.values())

    # Also pack legend-friendly pairs
    disease_data = list(zip(disease_labels, disease_values))


    
    
    

   
     # Prediction handling
    prediction = None
    if request.method == 'POST':
        # Collect form data
        sex = 1 if request.POST.get('SEX') == 'Male' else 2
        marstat = 1 if request.POST.get('MARSTAT') == 'Single' else 2
        cancer = 1 if request.POST.get('CANCEREV') == 'Yes' else 0
        chol = 1 if request.POST.get('CHOLHIGHEV') == 'Yes' else 0
        diabetes = 1 if request.POST.get('DIABETICEV') == 'Yes' else 0
        heart = 1 if request.POST.get('HEARTCONEV') == 'Yes' else 0
        hypertension = 1 if request.POST.get('HYPERTEN') == 'Yes' else 0
        hypertension_age = int(request.POST.get('HYPERTENAGE', 0))
        age = int(request.POST.get('AGE', 0))
        ins_type_form = request.POST.get('INSURANCE_TYPE')
        family_income = float(request.POST.get('FAMILY_INCOME', 0))
        family_size = int(request.POST.get('FAMILY_SIZE', 1))
        # target_insurance_type = request.POST.get('INSURANCE_TYPE')
        # Encode insurance type
        ins_map = {'Private Only': 0, 'Public Only': 1, 'Uninsured': 2, 'Mixed':3}
        ins_encoded = ins_map.get(ins_type_form, 2)

        insurance_type_encoded = (
            0 if ins_type_form == 'Private Only' else
            1 if ins_type_form == 'Public Only' else
            2 if ins_type_form == 'Uninsured' else 3 )  

        filtered_df = cleaned_df[cleaned_df['INSURANCE_TYPE'] == ins_type_form].copy()
       
         # OOP Burden
        avg_oop_burden = round(filtered_df['OOP_BURDEN'].mean() * 100, 2)

        # Catastrophic OOP Rate
        catastrophic_rate = round(filtered_df['CATASTROPHIC_OOP'].mean() * 100, 2)
        
        
        # AVG Charges/head
        if not filtered_df.empty:
                avg_charges_per_head = filtered_df['CHGTOT'].mean()
        else:
            avg_charges_per_head = 0

        # avg heaklth score
        cleaned_filtered_df = balanced_df[balanced_df['INSURANCE_TYPE_ENC'] == insurance_type_encoded].copy()
        avg_health_df = (
    cleaned_filtered_df.groupby('INSURANCE_TYPE_ENC')['HEALTH']
    .mean()
    .reset_index())

        if not avg_health_df.empty:
            match = avg_health_df.loc[
                avg_health_df['INSURANCE_TYPE_ENC'] == insurance_type_encoded, 'HEALTH'
            ]
            avg_health_score = float(match.values[0]) if not match.empty else 0.0
        else:
            avg_health_score = 0.0


        # ispoverty
    # Calculate poverty ratio & poverty flag
        poverty_threshold = get_poverty_threshold(family_size)
        poverty_ratio = family_income / poverty_threshold
        is_poverty = 1 if poverty_ratio < 1.0 else 0
        

        # Prepare features for model
        features = np.array([[sex, marstat, family_size, family_income, age, cancer, chol,
                              diabetes, heart, hypertension, is_poverty, ins_encoded]]) # change is poverty here instead of hypertenage
        

        if ins_type_form == "Private Only":
            ins_type_display = 'Private'
        elif ins_type_form == "Public Only":
            ins_type_display = 'Public'
        elif ins_type_form == "Mixed":
            ins_type_display = 'Mixed'
        else:
            ins_type_display = 'Uninsured'
        
        # Predict probability
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, 'dashboard', 'model', 'my_trained_model.pkl')
        model = pickle.load(open(MODEL_PATH, 'rb'))

        prediction = model.predict_proba(features)[0][1] * 100
        # Make prediction
        prediction = model.predict(features)[0]  # 0 = bad, 1 = good
        prediction_prob = model.predict_proba(features)[0][prediction] * 100
        predicted_health_status = "Good" if prediction == 1 else "Bad"
        suitability_status = "Suitable" if predicted_health_status == "Good" else "Not Suitable"
       
    print(f"disease_values: {disease_values  }")
    
    # ___________________________________________________________________________________________
    # chart data

    bins = [0, 18, 30, 45, 60, 75, 100]
    labels = ["0-18", "19-30", "31-45", "46-60", "61-75", "76+"]

    # # Copy dataset and add Age_Group + Health_Category
    # chart_df = balanced_df.copy()
    # chart_df["Age_Group"] = pd.cut(chart_df["new_age"], bins=bins, labels=labels, right=True)

    # # Health_Category: Good (1) vs Bad (0)
    # chart_df["Health_Category"] = chart_df["HEALTH"].apply(lambda x: "Good" if x <= 3 else "Bad")

   
    # chart_df["INSURANCE_LABEL"] = chart_df["INSURANCE_TYPE_ENC"]
    # # Group by insurance, age, health category
    # grouped = chart_df.groupby(["INSURANCE_LABEL", "Age_Group", "Health_Category"]).size().reset_index(name="count")

    # # Convert to nested dict structure: chart_data[insurance][age][health]
    # chart_data = {}
    # for ins in grouped["INSURANCE_LABEL"].unique():
    #     chart_data[ins] = {}
    #     for age in labels:
    #         chart_data[ins][str(age)] = {"Good": 0, "Bad": 0}

    # for _, row in grouped.iterrows():
    #     ins, age, health, count = row
    #     chart_data[ins][str(age)][health] = int(count)
    # Start with balanced_df
    chart_df = balanced_df.copy()

    chart_df["Age_Group"] = pd.cut(chart_df["new_age"], bins=bins, labels=labels, right=True)

    # Health Category: Good (1-3) vs Bad (4-5) -- adjust if your HEALTH is coded differently
    chart_df["Health_Category"] = chart_df["HEALTH"].apply(lambda x: "Good" if x ==1 else "Bad")
    # Mapping ENC → Label
    insurance_mapping = {
        0: "Private Only",
        1: "Public Only",
        2: "Uninsured", 
        3: "Mixed"
    }

        # Add a label column
    chart_df["INSURANCE_LABEL"] = chart_df["INSURANCE_TYPE_ENC"].map(insurance_mapping)

    # Group by insurance, age, health
    grouped = chart_df.groupby(["INSURANCE_LABEL", "Age_Group", "Health_Category"]).size().reset_index(name="count")

    # Convert into nested dict structure
    chart_data = {}
    for ins in grouped["INSURANCE_LABEL"].unique():
        chart_data[str(ins)] = {}   # <-- cast insurance type to str
        for age in labels:
            chart_data[str(ins)][str(age)] = {"Good": 0 , "Bad": 0}

    for _, row in grouped.iterrows():
        ins, age, health, count = row
        chart_data[str(ins)][str(age)][health] = int(count)


        # ---- Final: send to template ----




    context = {
        'avg_expenditure': avg_expenditure,
        'insurance_type': ins_type_display,
        'avg_charges_per_head': f"${avg_charges_per_head:,.2f}",
        "avg_oop_burden": avg_oop_burden,
        "catastrophic_rate": catastrophic_rate,
        'health_status_counts': all_health_data[ins_type_form],
        'all_health_data': json.dumps(all_health_data),
        'insurance_types': list(all_health_data.keys()),
        'avg_health_score':avg_health_score,
        "prediction": prediction_prob,
        "predicted_health_status": predicted_health_status,
        "predicted_insurance_type": request.POST.get("INSURANCE_TYPE") if request.method == "POST" else None,
        "suitability_status" : suitability_status,
        "disease_data": disease_data,
        "disease_labels": disease_labels,
        "disease_values": disease_values,
        "chart_data": json.dumps(chart_data),
        "age_groups": json.dumps(labels),  
        
    }

    return render(request, "index.html", context)


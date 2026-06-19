# Feature value mappings for the German Credit Dataset (Hofmann, 1994)
# Based on the original UCI dataset documentation (german.doc)

FEATURE_MAPPINGS = {
    "checking_account": {
        0: "No Checking Account",
        1: "< 0 DM",
        2: "0 - 200 DM",
        3: ">= 200 DM"
    },
    "credit_history": {
        0: "No Credits / All Paid",
        1: "All Credits Paid Back",
        2: "Existing Credits Paid Back",
        3: "Delay in Past",
        4: "Critical Account"
    },
    "purpose": {
        0: "Car (New)",
        1: "Car (Used)",
        2: "Furniture / Equipment",
        3: "Radio / TV",
        4: "Domestic Appliances",
        5: "Repairs",
        6: "Education",
        7: "Vacation",
        8: "Retraining",
        9: "Business",
        10: "Other"
    },
    "savings_account": {
        0: "No Savings Account",
        1: "< 100 DM",
        2: "100 - 500 DM",
        3: "500 - 1000 DM",
        4: ">= 1000 DM"
    },
    "employment": {
        0: "Unemployed",
        1: "< 1 Year",
        2: "1 - 4 Years",
        3: "4 - 7 Years",
        4: ">= 7 Years"
    },
    "personal_status": {
        0: "Male: Divorced / Separated",
        1: "Female: Divorced / Separated / Married",
        2: "Male: Single",
        3: "Male: Married / Widowed",
        4: "Female: Single"
    },
    "other_debtors": {
        0: "None",
        1: "Co-Applicant",
        2: "Guarantor"
    },
    "property": {
        0: "Real Estate",
        1: "Life Insurance / Savings",
        2: "Car / Other",
        3: "No Property"
    },
    "other_installments": {
        0: "Bank",
        1: "Stores",
        2: "None"
    },
    "housing": {
        0: "Rent",
        1: "Own",
        2: "For Free"
    },
    "job": {
        0: "Unemployed / Unskilled Non-Resident",
        1: "Unskilled Resident",
        2: "Skilled Employee",
        3: "Management / Self-Employed"
    },
    "telephone": {
        0: "No",
        1: "Yes"
    },
    "foreign_worker": {
        0: "No",
        1: "Yes"
    }
}

# Features displayed as numbers (continuous values)
NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "liable_people"
]

# Human-readable feature labels
FEATURE_LABELS = {
    "checking_account": "Checking Account",
    "duration": "Duration (months)",
    "credit_history": "Credit History",
    "purpose": "Purpose",
    "credit_amount": "Credit Amount (DM)",
    "savings_account": "Savings Account",
    "employment": "Employment",
    "installment_rate": "Installment Rate (%)",
    "personal_status": "Personal Status",
    "other_debtors": "Other Debtors",
    "residence_since": "Residence Since (years)",
    "property": "Property",
    "age": "Age (years)",
    "other_installments": "Other Installments",
    "housing": "Housing",
    "existing_credits": "Existing Credits",
    "job": "Job",
    "liable_people": "Liable People",
    "telephone": "Telephone",
    "foreign_worker": "Foreign Worker"
}


def decode_case(case):
    """Convert encoded feature values to human-readable labels."""
    decoded = {}
    for feature, value in case.items():
        label = FEATURE_LABELS.get(feature, feature)
        if feature in FEATURE_MAPPINGS:
            decoded[label] = FEATURE_MAPPINGS[feature].get(int(value), str(value))
        else:
            decoded[label] = value
    return decoded
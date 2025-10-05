import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta

# Initialize Faker to generate fake data
fake = Faker()

# --- Configuration ---
NUM_EMPLOYEES = 200
NUM_PROJECTS = 40
START_DATE = date(2023, 1, 1)
END_DATE = date(2023, 12, 31)

# --- Define Realistic Data ---
ROLES = {
    "Analyst": {"min_salary": 60000, "max_salary": 80000, "billing_rate": 150},
    "Consultant": {"min_salary": 80000, "max_salary": 120000, "billing_rate": 250},
    "Senior Consultant": {"min_salary": 120000, "max_salary": 160000, "billing_rate": 350},
    "Manager": {"min_salary": 160000, "max_salary": 220000, "billing_rate": 450},
    "Senior Manager": {"min_salary": 220000, "max_salary": 300000, "billing_rate": 600},
    "Partner": {"min_salary": 300000, "max_salary": 500000, "billing_rate": 800},
}

GEOGRAPHIES = ["New York", "London", "San Francisco", "Chicago", "Singapore", "Frankfurt"]
INDUSTRIES = ["Technology", "Financial Services", "Healthcare", "Consumer Goods", "Energy", "Manufacturing"]
PROJECT_TYPES = ["Strategy", "Operations", "Technology Implementation", "Digital Transformation", "Mergers & Acquisitions", "Risk & Compliance"]

# --- Generate Employee Data ---
employees = []
for i in range(NUM_EMPLOYEES):
    role = random.choice(list(ROLES.keys()))
    employees.append({
        "Employee ID": 1000 + i,
        "Employee Name": fake.name(),
        "Role": role,
        "Base Salary": random.randint(ROLES[role]["min_salary"], ROLES[role]["max_salary"]),
        "Geography": random.choice(GEOGRAPHIES)
    })
employees_df = pd.DataFrame(employees)

# --- Generate Project Data ---
projects = []
for i in range(NUM_PROJECTS):
    client_name = fake.company()
    industry = random.choice(INDUSTRIES)
    project_type = random.choice(PROJECT_TYPES)
    projects.append({
        "Project ID": 5000 + i,
        "Project Name": f"{industry} {project_type} for {client_name}",
        "Client Name": client_name,
        "Industry": industry
    })
projects_df = pd.DataFrame(projects)

# --- Generate Project Management/PSA System Data ---
psa_data = []
date_range = [START_DATE + timedelta(days=x) for x in range((END_DATE - START_DATE).days + 1)]

# Assign employees to projects
project_assignments = {emp_id: random.sample(list(projects_df["Project ID"]), k=random.randint(1, 3)) for emp_id in employees_df["Employee ID"]}

for single_date in date_range:
    # Simulate lower activity on weekends
    if single_date.weekday() < 5:  # Monday to Friday
        for index, employee in employees_df.iterrows():
            # Simulate some employees not working on projects every day
            if random.random() > 0.1: # 90% chance of working on a project
                project_id = random.choice(project_assignments[employee["Employee ID"]])
                billable_hours = round(np.random.normal(loc=7.5, scale=1.5), 2)
                billable_hours = max(0, min(10, billable_hours)) # Clamp between 0 and 10

                non_billable_hours = round(np.random.normal(loc=1.5, scale=1), 2)
                non_billable_hours = max(0, min(4, non_billable_hours))

                total_hours = billable_hours + non_billable_hours
                utilization_rate = (billable_hours / 8) * 100 if total_hours > 0 else 0 # Assuming 8 available hours

                psa_data.append({
                    "Date": single_date,
                    "Employee ID": employee["Employee ID"],
                    "Project ID": project_id,
                    "Billable Hours": billable_hours,
                    "Non-Billable Hours": non_billable_hours,
                    "Total Hours": total_hours,
                    "Utilization Rate (%)": round(utilization_rate, 2)
                })

psa_df = pd.DataFrame(psa_data)


# --- Generate Financial System Data ---
financial_data = []

# Merge PSA data with employee and project data to get necessary details for financial calculations
merged_df = pd.merge(psa_df, employees_df, on="Employee ID")
merged_df = pd.merge(merged_df, projects_df, on="Project ID")

# Calculate daily project cost and revenue
merged_df["Daily Project Cost"] = (merged_df["Base Salary"] / 260) * (merged_df["Total Hours"] / 8) # Assuming 260 working days
merged_df["Daily Project Revenue"] = merged_df["Billable Hours"] * merged_df["Role"].apply(lambda x: ROLES[x]["billing_rate"])

# Aggregate data by month and project
merged_df['Month'] = merged_df['Date'].dt.to_period('M')
financial_df = merged_df.groupby(['Month', 'Project ID', 'Project Name', 'Client Name', 'Industry']).agg(
    Total_Project_Cost=('Daily Project Cost', 'sum'),
    Total_Project_Revenue=('Daily Project Revenue', 'sum')
).reset_index()

financial_df = financial_df.round(2)


# --- Display Sample Data ---
print("--- Employee Data ---")
print(employees_df.head())
print("\n--- Project Data ---")
print(projects_df.head())
print("\n--- PSA System Data (Sample) ---")
print(psa_df.head())
print("\n--- Financial System Data (Sample) ---")
print(financial_df.head())

# --- Save to CSV (Optional) ---
# employees_df.to_csv("employees.csv", index=False)
# projects_df.to_csv("projects.csv", index=False)
# psa_df.to_csv("psa_data.csv", index=False)
# financial_df.to_csv("financial_data.csv", index=False)


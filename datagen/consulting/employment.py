import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker for generating realistic data
fake = Faker()

# --- Model Configuration ---

# Job Titles and their distribution
JOB_TITLES = {
    'Analyst': 0.35,
    'Senior Analyst': 0.20,
    'Consultant': 0.25,
    'Manager': 0.10,
    'Senior Manager': 0.05,
    'Associate Director': 0.03,
    'Managing Director': 0.02
}

# Departments and their distribution
DEPARTMENTS = {
    'Technology': 0.45,
    'Operations': 0.35,
    'Strategy & Consulting': 0.10,
    'Accenture Song': 0.05,
    'Industry X': 0.05
}

# Office Locations with a simple location-based pay adjustment factor
OFFICE_LOCATIONS = {
    'New York, NY': 1.15,
    'Chicago, IL': 1.05,
    'San Francisco, CA': 1.25,
    'Atlanta, GA': 1.0,
    'London, UK': 1.1,
    'Dublin, IE': 1.0,
    'Bengaluru, IN': 0.4,
    'Tokyo, JP': 1.1
}

# Employment Status distribution
EMPLOYMENT_STATUS_DIST = {'Full-time': 0.95, 'Contractor': 0.045, 'Part-time': 0.005}

# Annual Salary Ranges by Job Title (Base Salary)
SALARY_RANGES = {
    'Analyst': (70000, 90000),
    'Senior Analyst': (85000, 110000),
    'Consultant': (110000, 150000),
    'Manager': (140000, 180000),
    'Senior Manager': (170000, 220000),
    'Associate Director': (200000, 250000),
    'Managing Director': (250000, 400000)
}

# Annual Attrition Rate
ANNUAL_ATTRITION_RATE = 0.18

def generate_realistic_employment_data(begin_date, total_employees_begin, end_date, total_employees_end):
    """
    Generates a realistic employment dataset for a large consulting firm.

    Args:
        begin_date (str): The start date for the data generation (YYYY-MM-DD).
        total_employees_begin (int): The total number of employees on the begin date.
        end_date (str): The end date for the data generation (YYYY-MM-DD).
        total_employees_end (int): The total number of employees on the end date.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated employment data.
    """
    begin_date = datetime.strptime(begin_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    duration_days = (end_date - begin_date).days
    duration_years = duration_days / 365.25

    data = []

    # --- Generate Initial Employee Population ---
    for _ in range(total_employees_begin):
        # Assign Job Title and Department based on distribution
        job_title = np.random.choice(list(JOB_TITLES.keys()), p=list(JOB_TITLES.values()))
        department = np.random.choice(list(DEPARTMENTS.keys()), p=list(DEPARTMENTS.values()))

        # Assign Office Location and get pay adjustment
        office, pay_adj = random.choice(list(OFFICE_LOCATIONS.items()))

        # Assign Employment Status
        employment_status = np.random.choice(list(EMPLOYMENT_STATUS_DIST.keys()), p=list(EMPLOYMENT_STATUS_DIST.values()))

        # Generate Hire Date
        hire_date = fake.date_between(start_date='-10y', end_date=begin_date)

        # Determine Pay Rate
        if employment_status == 'Contractor':
            base_salary = random.randint(*SALARY_RANGES[job_title]) * pay_adj
            hourly_rate = round((base_salary / 2080) * 1.25, 2) # Assume 2080 work hours/year and a 25% premium for contractors
            pay_rate = f"${hourly_rate}/hour"
            pay_frequency = 'Weekly'
        else:
            salary = random.randint(*SALARY_RANGES[job_title]) * pay_adj
            pay_rate = f"${int(salary):,}"
            pay_frequency = 'Semi-monthly' if office.endswith('US') else 'Monthly'


        employee_data = {
            'Job Title/Position': job_title,
            'Department/Business Unit': department,
            'Office Location/Work Address': office,
            'Employment Status': employment_status,
            'Hire Date': hire_date,
            'Termination Date': None,
            'Pay Group/Pay Frequency': pay_frequency,
            'Pay Rate': pay_rate,
            'Exemption Status': 'Exempt' # Consulting roles are typically exempt
        }
        data.append(employee_data)

    df = pd.DataFrame(data)

    # --- Simulate Changes Between Begin and End Date ---
    net_change = total_employees_end - total_employees_begin
    num_terminations = int(total_employees_begin * ANNUAL_ATTRITION_RATE * duration_years)
    num_hires = num_terminations + net_change

    # Process Terminations
    termination_indices = df.sample(n=min(num_terminations, len(df))).index
    for idx in termination_indices:
        hire_date = df.loc[idx, 'Hire Date']
        if isinstance(hire_date, str):
            hire_date = datetime.strptime(hire_date, '%Y-%m-%d').date()
        termination_date = fake.date_between_dates(date_start=max(begin_date.date(), hire_date + timedelta(days=90)), date_end=end_date.date())
        df.loc[idx, 'Termination Date'] = termination_date

    # Process New Hires
    new_hires = []
    for _ in range(num_hires):
        job_title = np.random.choice(list(JOB_TITLES.keys()), p=list(JOB_TITLES.values()))
        department = np.random.choice(list(DEPARTMENTS.keys()), p=list(DEPARTMENTS.values()))
        office, pay_adj = random.choice(list(OFFICE_LOCATIONS.items()))
        employment_status = np.random.choice(list(EMPLOYMENT_STATUS_DIST.keys()), p=list(EMPLOYMENT_STATUS_DIST.values()))
        hire_date = fake.date_between_dates(date_start=begin_date, date_end=end_date)

        if employment_status == 'Contractor':
            base_salary = random.randint(*SALARY_RANGES[job_title]) * pay_adj
            hourly_rate = round((base_salary / 2080) * 1.25, 2)
            pay_rate = f"${hourly_rate}/hour"
            pay_frequency = 'Weekly'
        else:
            salary = random.randint(*SALARY_RANGES[job_title]) * pay_adj
            pay_rate = f"${int(salary):,}"
            pay_frequency = 'Semi-monthly' if office.endswith('US') else 'Monthly'

        new_hire_data = {
            'Job Title/Position': job_title,
            'Department/Business Unit': department,
            'Office Location/Work Address': office,
            'Employment Status': employment_status,
            'Hire Date': hire_date,
            'Termination Date': None,
            'Pay Group/Pay Frequency': pay_frequency,
            'Pay Rate': pay_rate,
            'Exemption Status': 'Exempt'
        }
        new_hires.append(new_hire_data)

    df = pd.concat([df, pd.DataFrame(new_hires)], ignore_index=True)

    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate realistic employment data for a large consulting firm.")
    parser.add_argument("--begin_date", type=str, help="The start date for the data generation (YYYY-MM-DD).")
    parser.add_argument("--total_employees_begin", type=int, help="The total number of employees on the begin date.")
    parser.add_argument("--end_date", type=str, help="The end date for the data generation (YYYY-MM-DD).")
    parser.add_argument("--total_employees_end", type=int, help="The total number of employees on the end date.")
    args = parser.parse_args()

    generated_df = generate_realistic_employment_data(
        args.begin_date,
        args.total_employees_begin,
        args.end_date,
        args.total_employees_end
    )
    generated_df.to_csv('realistic_employment_data.csv', index=False)
    print("Successfully generated 'realistic_employment_data.csv'")


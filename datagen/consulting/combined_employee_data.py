import pandas as pd
import numpy as np
from faker import Faker
from datetime import date, timedelta, datetime
import random
import argparse

# Initialize Faker for US English
fake = Faker('en_US')

# --- CONFIGURATION PARAMETERS ---

# 1. General Settings
NUM_EMPLOYEES = 1000  # Total number of employees to generate.
OUTPUT_FILE = 'combined_employee_data.csv'

# 2. Location Settings (US-based example)
OFFICE_LOCATION = {
    'city': 'Chicago',
    'state': 'IL',
    'zipcode_prefix': '606' # Helps generate more relevant local zip codes
}
# List of major cities for remote workers
REMOTE_CITIES = [
    {'city': 'New York', 'state': 'NY'}, {'city': 'Los Angeles', 'state': 'CA'},
    {'city': 'Houston', 'state': 'TX'}, {'city': 'Phoenix', 'state': 'AZ'},
    {'city': 'Philadelphia', 'state': 'PA'}, {'city': 'San Antonio', 'state': 'TX'},
    {'city': 'San Diego', 'state': 'CA'}, {'city': 'Dallas', 'state': 'TX'},
    {'city': 'San Jose', 'state': 'CA'}, {'city': 'Austin', 'state': 'TX'}
]
REMOTE_PERCENTAGE = 0.20  # 20% of employees work remotely.

# 3. Demographic Settings (Based on general US statistics, you can adjust)
GENDER_DISTRIBUTION = {'Male': 0.50, 'Female': 0.49, 'Non-Binary': 0.01}
MARITAL_STATUS_DISTRIBUTION = {'Single': 0.40, 'Married': 0.50, 'Divorced': 0.10}
# Age distribution: (min_age, max_age, weight)
AGE_DISTRIBUTION = [
    (18, 25, 0.15),  # 15% of employees are between 18-25
    (26, 35, 0.35),  # 35% between 26-35
    (36, 45, 0.25),  # 25% between 36-45
    (46, 55, 0.15),  # 15% between 46-55
    (56, 65, 0.10)   # 10% between 56-65
]

# 4. Employment Configuration
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

# --- HELPER FUNCTIONS ---

def get_weighted_choice(distribution):
    """Helper function to make a weighted choice from a dictionary."""
    choices = list(distribution.keys())
    weights = list(distribution.values())
    return np.random.choice(choices, p=weights)

def generate_dob_from_age_distribution():
    """Generates a date of birth based on the weighted age distribution."""
    age_ranges = [item[0:2] for item in AGE_DISTRIBUTION]
    weights = [item[2] for item in AGE_DISTRIBUTION]
    
    selected_range_index = np.random.choice(len(age_ranges), p=weights)
    min_age, max_age = age_ranges[selected_range_index]
    
    age = random.randint(min_age, max_age)
    today = date.today()
    birth_year = today.year - age
    # Random day within the year
    start_of_year = date(birth_year, 1, 1)
    end_of_year = date(birth_year, 12, 31)
    time_between_dates = end_of_year - start_of_year
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    
    return start_of_year + timedelta(days=random_number_of_days)

def create_combined_employee_record(employee_id, existing_ssns):
    """Creates a single, combined employee record with both personal and employment details."""
    
    # --- PERSONAL DEMOGRAPHICS ---
    gender = get_weighted_choice(GENDER_DISTRIBUTION)
    marital_status = get_weighted_choice(MARITAL_STATUS_DISTRIBUTION)
    
    if gender == 'Male':
        first_name = fake.first_name_male()
        middle_initial = fake.random_uppercase_letter()
    elif gender == 'Female':
        first_name = fake.first_name_female()
        middle_initial = fake.random_uppercase_letter()
    else:
        first_name = fake.first_name_nonbinary()
        middle_initial = fake.random_uppercase_letter()
        
    last_name = fake.last_name()
    email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1,99)}@{fake.free_email_domain()}"

    # Unique Identifiers
    while True:
        ssn = fake.ssn()
        if ssn not in existing_ssns:
            existing_ssns.add(ssn)
            break
            
    # Date of Birth
    dob = generate_dob_from_age_distribution()

    # Contact and Address
    phone_number = fake.phone_number()
    
    # --- EMPLOYMENT DETAILS ---
    # Assign Job Title and Department based on distribution
    job_title = get_weighted_choice(JOB_TITLES)
    department = get_weighted_choice(DEPARTMENTS)

    # Assign Office Location and get pay adjustment
    office_location, pay_adj = random.choice(list(OFFICE_LOCATIONS.items()))

    # Assign Employment Status
    employment_status = get_weighted_choice(EMPLOYMENT_STATUS_DIST)

    # Generate Hire Date (within last 10 years)
    hire_date = fake.date_between(start_date='-10y', end_date='today')

    # Determine Pay Rate
    if employment_status == 'Contractor':
        base_salary = random.randint(*SALARY_RANGES[job_title]) * pay_adj
        hourly_rate = round((base_salary / 2080) * 1.25, 2) # Assume 2080 work hours/year and a 25% premium for contractors
        pay_rate = f"${hourly_rate}/hour"
        pay_frequency = 'Weekly'
    else:
        salary = random.randint(*SALARY_RANGES[job_title]) * pay_adj
        pay_rate = f"${int(salary):,}"
        pay_frequency = 'Semi-monthly' if office_location.endswith('US') or office_location.endswith('NY') or office_location.endswith('CA') or office_location.endswith('IL') or office_location.endswith('GA') or office_location.endswith('TX') or office_location.endswith('AZ') or office_location.endswith('PA') else 'Monthly'

    # Determine if local or remote for personal address
    if random.random() < REMOTE_PERCENTAGE:
        # Generate remote employee address
        location = random.choice(REMOTE_CITIES)
        street_address = fake.street_address()
        personal_city = location['city']
        personal_state = location['state']
        zip_code = fake.zipcode_in_state(personal_state)
    else:
        # Generate local employee address
        street_address = fake.street_address()
        personal_city = OFFICE_LOCATION['city']
        personal_state = OFFICE_LOCATION['state']
        # Generate a zip code starting with the office prefix for higher locality
        zip_code = OFFICE_LOCATION['zipcode_prefix'] + str(random.randint(10, 99))

    # Determine termination date (18% annual attrition rate)
    termination_date = None
    if random.random() < (ANNUAL_ATTRITION_RATE * (datetime.now().date() - hire_date).days / 365.25):
        # Employee has been terminated
        termination_date = fake.date_between(start_date=hire_date + timedelta(days=90), end_date='today')

    return {
        # Personal Information
        'Employee ID': f"EMP{employee_id:05d}",
        'First Name': first_name,
        'Middle Initial': middle_initial,
        'Last Name': last_name,
        'SSN': ssn,
        'Date of Birth': dob,
        'Gender': gender,
        'Marital Status': marital_status,
        'Street Address': street_address,
        'Personal City': personal_city,
        'Personal State': personal_state,
        'Zip Code': zip_code,
        'Phone Number': phone_number,
        'Email': email,
        
        # Employment Information
        'Job Title': job_title,
        'Department': department,
        'Office Location': office_location,
        'Employment Status': employment_status,
        'Hire Date': hire_date,
        'Termination Date': termination_date,
        'Pay Frequency': pay_frequency,
        'Pay Rate': pay_rate,
        'Exemption Status': 'Exempt'  # Consulting roles are typically exempt
    }

def generate_combined_employee_data(num_employees):
    """Generates combined employee data with both personal and employment details."""
    print(f"Generating {num_employees} combined employee records...")
    
    employees = []
    existing_ssns = set()  # To ensure SSNs are unique

    for i in range(1, num_employees + 1):
        employees.append(create_combined_employee_record(i, existing_ssns))
        if i % 100 == 0:
            print(f"Generated {i} records...")

    return pd.DataFrame(employees)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate combined employee and employment data.")
    parser.add_argument("--num_employees", type=int, default=NUM_EMPLOYEES, 
                       help=f"Number of employees to generate (default: {NUM_EMPLOYEES})")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                       help=f"Output CSV file name (default: {OUTPUT_FILE})")
    
    args = parser.parse_args()
    
    # Generate the data
    df = generate_combined_employee_data(args.num_employees)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    print(f"\nSuccessfully generated {args.num_employees} combined employee records.")
    print(f"Data saved to '{args.output}'")
    print(f"\n--- Sample Data ---")
    print(df.head())
    print(f"\n--- Data Summary ---")
    print(f"Total records: {len(df)}")
    print(f"Active employees: {len(df[df['Termination Date'].isna()])}")
    print(f"Terminated employees: {len(df[df['Termination Date'].notna()])}")
    print(f"\nDepartment distribution:")
    print(df['Department'].value_counts())
    print(f"\nJob title distribution:")
    print(df['Job Title'].value_counts())

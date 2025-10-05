import pandas as pd
import numpy as np
from faker import Faker
from datetime import date, timedelta
import random

# --- CONFIGURATION PARAMETERS ---
# You can change these values to control the output.

# 1. General Settings
NUM_EMPLOYEES = 1000  # Total number of employees to generate.
OUTPUT_FILE = 'employee_data.csv'

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

# --- DATA GENERATION LOGIC ---

# Initialize Faker for US English
fake = Faker('en_US')

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

def create_employee_record(employee_id, existing_ssns):
    """Creates a single, unique employee record."""
    
    # 1. Demographics
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

    # 2. Unique Identifiers
    while True:
        ssn = fake.ssn()
        if ssn not in existing_ssns:
            existing_ssns.add(ssn)
            break
            
    # 3. Date of Birth
    dob = generate_dob_from_age_distribution()

    # 4. Contact and Address
    phone_number = fake.phone_number()
    
    # Determine if local or remote
    if random.random() < REMOTE_PERCENTAGE:
        # Generate remote employee address
        location = random.choice(REMOTE_CITIES)
        street_address = fake.street_address()
        city = location['city']
        state = location['state']
        zip_code = fake.zipcode_in_state(state)
    else:
        # Generate local employee address
        street_address = fake.street_address()
        city = OFFICE_LOCATION['city']
        state = OFFICE_LOCATION['state']
        # Generate a zip code starting with the office prefix for higher locality
        zip_code = OFFICE_LOCATION['zipcode_prefix'] + str(random.randint(10, 99))

    return {
        'Employee ID': f"EMP{employee_id:05d}",
        'First Name': first_name,
        'Middle Initial': middle_initial,
        'Last Name': last_name,
        'SSN': ssn,
        'Date of Birth': dob,
        'Gender': gender,
        'Marital Status': marital_status,
        'Street Address': street_address,
        'City': city,
        'State': state,
        'Zip Code': zip_code,
        'Phone Number': phone_number,
        'Email': email
    }

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Generating employee data...")
    
    employees = []
    existing_ssns = set() # To ensure SSNs are unique

    for i in range(1, NUM_EMPLOYEES + 1):
        employees.append(create_employee_record(i, existing_ssns))

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(employees)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Successfully generated {NUM_EMPLOYEES} employee records.")
    print(f"Data saved to '{OUTPUT_FILE}'")
    print("\n--- Sample Data ---")
    print(df.head())


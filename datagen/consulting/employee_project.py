import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List

def generate_consulting_firm_data(
    start_date: str,
    end_date: str,
    start_employee_count: int,
    end_employee_count: int,
    geos: List[str]
):
    """
    Generate realistic consulting firm dataset with employees, projects, and assignments.
    
    Parameters:
    - start_date: str (e.g., "2023-01-01")
    - end_date: str (e.g., "2025-12-31")
    - start_employee_count: int
    - end_employee_count: int
    - geos: list[str] (e.g., ["US", "India", "UK"])
    """
    np.random.seed(42)
    random.seed(42)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days
    
    # Consulting firm title distribution (pyramid structure)
    title_distribution = {
        "Analyst": 0.42,
        "Consultant": 0.28,
        "Manager": 0.17,
        "Senior Manager": 0.09,
        "Director": 0.03,
        "Partner": 0.01
    }
    
    # Geo distribution (typical for global consulting firms)
    if len(geos) == 1:
        geo_distribution = {geos[0]: 1.0}
    elif len(geos) == 2:
        geo_distribution = {geos[0]: 0.65, geos[1]: 0.35}
    elif len(geos) == 3:
        geo_distribution = {geos[0]: 0.60, geos[1]: 0.25, geos[2]: 0.15}
    else:
        # For more geos, distribute with decreasing weights
        weights = np.array([0.6, 0.25, 0.10] + [0.05 / (len(geos) - 3)] * (len(geos) - 3))
        weights = weights / weights.sum()
        geo_distribution = {geo: weight for geo, weight in zip(geos, weights)}
    
    # Industry and project type distributions
    industries = ["Finance", "Healthcare", "Technology", "Retail", "Manufacturing", 
                  "Energy", "Telecom", "Public Sector"]
    project_types = ["Digital Transformation", "Salesforce Implementation", "Data Migration",
                     "Strategy Consulting", "Cloud Migration", "ERP Implementation",
                     "Business Process Optimization", "M&A Advisory", "Risk Management"]
    
    # ==================== GENERATE EMPLOYEES ====================
    employees = []
    employee_id = 1000
    
    # Calculate monthly employee counts for smooth growth
    months = pd.date_range(start_dt, end_dt, freq='MS')
    employee_counts = np.linspace(start_employee_count, end_employee_count, len(months))
    employee_counts = employee_counts.astype(int)
    
    # Generate initial employees
    for _ in range(start_employee_count):
        title = np.random.choice(list(title_distribution.keys()), 
                                p=list(title_distribution.values()))
        geo = np.random.choice(list(geo_distribution.keys()), 
                              p=list(geo_distribution.values()))
        
        # Hire date is before or at start date
        hire_date = start_dt - timedelta(days=random.randint(0, 1825))  # up to 5 years before
        
        # Utilization rate varies by title
        utilization_rates = {
            "Analyst": (0.75, 0.90),
            "Consultant": (0.80, 0.95),
            "Manager": (0.70, 0.85),
            "Senior Manager": (0.60, 0.75),
            "Director": (0.50, 0.65),
            "Partner": (0.40, 0.55)
        }
        util_min, util_max = utilization_rates[title]
        utilization_rate = round(random.uniform(util_min, util_max), 2)
        
        employees.append({
            "employee_id": employee_id,
            "name": f"Employee_{employee_id}",
            "title": title,
            "geo": geo,
            "hire_date": hire_date.strftime("%Y-%m-%d"),
            "exit_date": None,
            "utilization_rate": utilization_rate,
            "vacation_days_per_year": random.choice([15, 20, 25, 30])
        })
        employee_id += 1
    
    # Track active employees over time for growth and attrition
    active_employees = list(range(1000, 1000 + start_employee_count))
    current_month_idx = 0
    
    for month_dt in months[1:]:
        current_month_idx += 1
        target_count = employee_counts[current_month_idx]
        current_count = len(active_employees)
        
        # Annual attrition rate of 10-15%
        monthly_attrition_rate = 0.12 / 12
        num_exits = int(current_count * monthly_attrition_rate)
        
        # Some employees exit
        if num_exits > 0 and len(active_employees) > 0:
            exiting_ids = random.sample(active_employees, min(num_exits, len(active_employees)))
            for emp_id in exiting_ids:
                # Set exit date
                for emp in employees:
                    if emp["employee_id"] == emp_id and emp["exit_date"] is None:
                        exit_date = month_dt + timedelta(days=random.randint(0, 28))
                        if exit_date <= end_dt:
                            emp["exit_date"] = exit_date.strftime("%Y-%m-%d")
                active_employees.remove(emp_id)
        
        # Hire new employees to reach target
        num_hires = target_count - len(active_employees)
        for _ in range(max(0, num_hires)):
            title = np.random.choice(list(title_distribution.keys()), 
                                    p=list(title_distribution.values()))
            geo = np.random.choice(list(geo_distribution.keys()), 
                                  p=list(geo_distribution.values()))
            hire_date = month_dt + timedelta(days=random.randint(0, 28))
            
            if hire_date <= end_dt:
                util_min, util_max = utilization_rates[title]
                utilization_rate = round(random.uniform(util_min, util_max), 2)
                
                employees.append({
                    "employee_id": employee_id,
                    "name": f"Employee_{employee_id}",
                    "title": title,
                    "geo": geo,
                    "hire_date": hire_date.strftime("%Y-%m-%d"),
                    "exit_date": None,
                    "utilization_rate": utilization_rate,
                    "vacation_days_per_year": random.choice([15, 20, 25, 30])
                })
                active_employees.append(employee_id)
                employee_id += 1
    
    employees_df = pd.DataFrame(employees)
    
    # ==================== GENERATE PROJECTS ====================
    projects = []
    project_id = 5000
    
    # Generate projects throughout the time period
    # Average 1 new project per 8-10 employees per year
    num_projects = int((start_employee_count + end_employee_count) / 2 / 8 * (total_days / 365))
    
    for _ in range(num_projects):
        # Project start distributed throughout the period
        project_start = start_dt + timedelta(days=random.randint(0, max(1, total_days - 180)))
        
        # Project duration: 3-18 months, with most in 3-6 month range
        duration_weights = [0.35, 0.30, 0.20, 0.10, 0.03, 0.02]  # 3, 4-6, 7-9, 10-12, 13-15, 16-18 months
        duration_ranges = [(90, 120), (120, 180), (180, 270), (270, 365), (365, 455), (455, 545)]
        duration_range = random.choices(duration_ranges, weights=duration_weights)[0]
        duration_days = random.randint(*duration_range)
        project_end = project_start + timedelta(days=duration_days)
        
        # Don't extend beyond end_date
        if project_end > end_dt:
            project_end = end_dt
        
        # Determine project status based on dates
        today = datetime.now()
        if project_start > today:
            status = "Planned"
        elif project_end < today:
            status = "Completed"
        else:
            status = random.choice(["Active", "Active", "Active", "On Hold"])
        
        billing_model = np.random.choice(["Time-and-Materials", "Fixed-Price", "Retainer"],
                                        p=[0.60, 0.30, 0.10])
        
        projects.append({
            "project_id": project_id,
            "name": f"Project_{project_id}",
            "client_industry": random.choice(industries),
            "type": random.choice(project_types),
            "status": status,
            "start_date": project_start.strftime("%Y-%m-%d"),
            "end_date": project_end.strftime("%Y-%m-%d"),
            "billing_model": billing_model,
            "geo": np.random.choice(list(geo_distribution.keys()), 
                                   p=list(geo_distribution.values()))
        })
        project_id += 1
    
    projects_df = pd.DataFrame(projects)
    
    # ==================== GENERATE ASSIGNMENTS ====================
    assignments = []
    assignment_id = 10000
    
    # Project roles distribution
    project_roles = ["Developer", "QA", "Architect", "PM", "Business Analyst", 
                    "Tech Lead", "Scrum Master", "Designer"]
    
    # Track employee daily allocations to prevent overallocation
    # Structure: {employee_id: {date: total_allocation_pct}}
    employee_daily_allocations = {}
    
    for _, project in projects_df.iterrows():
        proj_start = datetime.strptime(project["start_date"], "%Y-%m-%d")
        proj_end = datetime.strptime(project["end_date"], "%Y-%m-%d")
        proj_duration_days = (proj_end - proj_start).days
        
        # Determine team size based on project duration
        if proj_duration_days < 150:  # Small project (< 5 months)
            team_size = random.randint(3, 6)
        elif proj_duration_days < 270:  # Medium project (5-9 months)
            team_size = random.randint(7, 15)
        else:  # Large project (9+ months)
            team_size = random.randint(15, 40)
        
        # Get eligible employees (hired before project ends, active during project)
        eligible_employees = []
        for _, emp in employees_df.iterrows():
            emp_hire = datetime.strptime(emp["hire_date"], "%Y-%m-%d")
            emp_exit = datetime.strptime(emp["exit_date"], "%Y-%m-%d") if emp["exit_date"] else end_dt
            
            # Employee must be hired before project starts and not exit before project starts
            if emp_hire <= proj_start and emp_exit >= proj_start:
                eligible_employees.append(emp)
        
        if len(eligible_employees) == 0:
            continue
        
        # Select team members (prefer same geo, but allow others)
        same_geo_employees = [e for e in eligible_employees if e["geo"] == project["geo"]]
        other_geo_employees = [e for e in eligible_employees if e["geo"] != project["geo"]]
        
        # 70% same geo, 30% other geos
        num_same_geo = int(team_size * 0.7)
        num_other_geo = team_size - num_same_geo
        
        selected_employees = []
        if len(same_geo_employees) >= num_same_geo:
            selected_employees.extend(random.sample(same_geo_employees, num_same_geo))
        else:
            selected_employees.extend(same_geo_employees)
        
        remaining_needed = team_size - len(selected_employees)
        if remaining_needed > 0 and len(other_geo_employees) > 0:
            selected_employees.extend(random.sample(other_geo_employees, 
                                                   min(remaining_needed, len(other_geo_employees))))
        
        # Add more employees to improve utilization (allow employees to work on multiple projects)
        # This helps achieve better overall utilization
        additional_employees_needed = max(0, team_size * 2 - len(selected_employees))  # Try for 2x team size
        if additional_employees_needed > 0:
            # Get employees not yet selected who could contribute
            selected_employee_ids = {emp["employee_id"] for emp in selected_employees}
            remaining_eligible = [e for e in eligible_employees if e["employee_id"] not in selected_employee_ids]
            additional_employees = random.sample(
                remaining_eligible, 
                min(additional_employees_needed, len(remaining_eligible))
            )
            selected_employees.extend(additional_employees)
        
        # Ensure at least one senior person (Manager+)
        senior_titles = ["Manager", "Senior Manager", "Director", "Partner"]
        has_senior = any(emp["title"] in senior_titles for emp in selected_employees)
        if not has_senior and len(eligible_employees) > 0:
            senior_employees = [e for e in eligible_employees if e["title"] in senior_titles]
            if senior_employees:
                selected_employees[0] = random.choice(senior_employees)
        
        # Create assignments for selected employees with proper allocation management
        # Sort employees by utilization rate (higher first) to prioritize better utilization
        selected_employees.sort(key=lambda x: x["utilization_rate"], reverse=True)
        
        for emp in selected_employees:
            emp_hire = datetime.strptime(emp["hire_date"], "%Y-%m-%d")
            emp_exit = datetime.strptime(emp["exit_date"], "%Y-%m-%d") if emp["exit_date"] else end_dt
            
            # Assignment starts when employee is available and project has started
            assign_start = max(proj_start, emp_hire)
            
            # Assignment ends when project ends or employee exits
            assign_end = min(proj_end, emp_exit)
            
            # Some employees join/leave mid-project (20% chance)
            if random.random() < 0.20:
                # Join late
                late_days = random.randint(0, proj_duration_days // 3)
                assign_start = min(assign_start + timedelta(days=late_days), proj_end)
            
            if random.random() < 0.20:
                # Leave early
                early_days = random.randint(0, proj_duration_days // 3)
                assign_end = max(assign_end - timedelta(days=early_days), assign_start)
            
            # Calculate appropriate allocation percentage based on employee utilization rate
            # and existing allocations
            emp_id = emp["employee_id"]
            emp_util_rate = emp["utilization_rate"]
            
            # Determine allocation based on employee utilization rate
            # Try to use the full utilization rate, but allow multiple projects when possible
            target_allocation = int(emp_util_rate * 100)
            
            # Check what allocation is possible for this assignment period
            min_available_allocation = 100  # Start with full availability
            current_date = assign_start
            
            while current_date <= assign_end:
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Get current allocation for this employee on this date
                if emp_id not in employee_daily_allocations:
                    employee_daily_allocations[emp_id] = {}
                
                current_allocation = employee_daily_allocations[emp_id].get(date_str, 0)
                available_allocation = 100 - current_allocation
                min_available_allocation = min(min_available_allocation, available_allocation)
                
                current_date += timedelta(days=1)
            
            # Skip if no allocation is possible
            if min_available_allocation < 20:
                continue
            
            # Determine the best allocation percentage
            # Strategy: Be more aggressive about achieving target utilization
            if min_available_allocation >= target_allocation:
                actual_allocation = target_allocation
            else:
                # Use most of available space to maximize utilization
                actual_allocation = min_available_allocation
            
            # Round to common allocation percentages, but prefer higher allocations
            allocation_options = [20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
            
            # If we have good availability, prefer higher allocations
            if min_available_allocation >= 70:
                # Prefer higher allocations for better utilization
                preferred_options = [70, 80, 90, 100]
                actual_allocation = min(preferred_options, key=lambda x: abs(x - actual_allocation))
            else:
                # Use closest available option
                actual_allocation = min(allocation_options, key=lambda x: abs(x - actual_allocation))
            
            actual_allocation = min(actual_allocation, min_available_allocation)
            
            # Skip if allocation is too low
            if actual_allocation < 20:
                continue
            
            # Role based on title
            title_to_role = {
                "Analyst": ["Developer", "QA", "Business Analyst"],
                "Consultant": ["Developer", "Business Analyst", "Tech Lead"],
                "Manager": ["PM", "Tech Lead", "Scrum Master"],
                "Senior Manager": ["PM", "Architect"],
                "Director": ["PM", "Architect"],
                "Partner": ["PM"]
            }
            role_options = title_to_role.get(emp["title"], project_roles)
            role = random.choice(role_options)
            
            # Update employee daily allocations
            current_date = assign_start
            while current_date <= assign_end:
                date_str = current_date.strftime("%Y-%m-%d")
                if date_str not in employee_daily_allocations[emp_id]:
                    employee_daily_allocations[emp_id][date_str] = 0
                employee_daily_allocations[emp_id][date_str] += actual_allocation
                current_date += timedelta(days=1)
            
            assignments.append({
                "assignment_id": assignment_id,
                "employee_id": emp["employee_id"],
                "project_id": project["project_id"],
                "role_on_project": role,
                "start_date": assign_start.strftime("%Y-%m-%d"),
                "end_date": assign_end.strftime("%Y-%m-%d"),
                "allocation_pct": actual_allocation
            })
            assignment_id += 1
    

    assignments_df = pd.DataFrame(assignments)
    
    # ==================== SAVE TO CSV ====================
    employees_df.to_csv("/tmp/employees.csv", index=False)
    projects_df.to_csv("/tmp/projects.csv", index=False)
    assignments_df.to_csv("/tmp/assignments.csv", index=False)
    
    print(f"✓ Generated {len(employees_df)} employees")
    print(f"✓ Generated {len(projects_df)} projects")
    print(f"✓ Generated {len(assignments_df)} assignments")
    print(f"\nFiles saved to /tmp/")
    print(f"  - employees.csv")
    print(f"  - projects.csv")
    print(f"  - assignments.csv")
    
    return employees_df, projects_df, assignments_df


# Example usage
if __name__ == "__main__":
    employees_df, projects_df, assignments_df = generate_consulting_firm_data(
        start_date="2023-01-01",
        end_date="2025-12-31",
        start_employee_count=1000,
        end_employee_count=1350,
        geos=["India", "US", "UK", "Germany"]
    )
    
    # Display sample data
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    print("\nEmployees (first 5):")
    print(employees_df.head())
    print("\nProjects (first 5):")
    print(projects_df.head())
    print("\nAssignments (first 5):")
    print(assignments_df.head())
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nEmployee title distribution:")
    print(employees_df["title"].value_counts(normalize=True).round(3))
    print(f"\nEmployee geo distribution:")
    print(employees_df["geo"].value_counts(normalize=True).round(3))
    print(f"\nProject status distribution:")
    print(projects_df["status"].value_counts())
    print(f"\nBilling model distribution:")
    print(projects_df["billing_model"].value_counts(normalize=True).round(3))
    print(f"\nAverage team size per project: {len(assignments_df) / len(projects_df):.1f}")
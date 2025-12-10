"""
Consulting Business ERP Data Generator

Generates realistic ERP records for a mid-sized consulting firm with ~1000 employees.
Simulates employees, projects, time tracking, HR events, and payroll data.

Usage:
    python generate_consulting_erp.py --start_date 2024-01-01 --end_date 2024-12-31
                                      --start_employees 950 --end_employees 1050
"""

import argparse
import csv
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import uuid


# ============================================================================
# Constants and Configuration
# ============================================================================

DEPARTMENTS = [
    "Strategy", "Operations", "Technology", "Financial Services",
    "Healthcare", "Energy", "Retail", "HR", "Finance", "Marketing", "IT"
]

# Job hierarchy with levels, titles, and salary ranges (2025 data)
JOB_LEVELS = {
    1: {"titles": ["Business Analyst", "Associate"], "salary_range": (80000, 95000), "weight": 0.35},
    2: {"titles": ["Consultant", "Senior Associate"], "salary_range": (135000, 165000), "weight": 0.25},
    3: {"titles": ["Senior Consultant"], "salary_range": (165000, 195000), "weight": 0.18},
    4: {"titles": ["Manager", "Project Manager"], "salary_range": (180000, 230000), "weight": 0.12},
    5: {"titles": ["Senior Manager"], "salary_range": (220000, 280000), "weight": 0.06},
    6: {"titles": ["Principal", "Associate Partner"], "salary_range": (280000, 400000), "weight": 0.03},
    7: {"titles": ["Partner", "Vice President"], "salary_range": (400000, 650000), "weight": 0.008},
    8: {"titles": ["Senior Partner", "Managing Director"], "salary_range": (600000, 1000000), "weight": 0.002},
}

# Consulting project types
PROJECT_TYPES = [
    "Strategy Development", "Digital Transformation", "M&A Due Diligence",
    "Operational Excellence", "Technology Implementation", "Change Management",
    "Cost Reduction", "Market Entry", "Organizational Restructuring",
    "IT Strategy", "Business Process Optimization", "Risk Management"
]

# Skills by department
SKILLS_BY_DEPT = {
    "Strategy": ["Strategic Planning", "Market Analysis", "Business Development", "Financial Modeling"],
    "Operations": ["Process Improvement", "Supply Chain", "Lean Six Sigma", "Project Management"],
    "Technology": ["Cloud Computing", "Data Analytics", "AI/ML", "Cybersecurity", "ERP Systems"],
    "Financial Services": ["Financial Analysis", "Risk Management", "Regulatory Compliance", "Investment Banking"],
    "Healthcare": ["Healthcare Operations", "Clinical Operations", "Healthcare IT", "Regulatory Compliance"],
    "Energy": ["Energy Markets", "Sustainability", "Oil & Gas", "Renewable Energy"],
    "Retail": ["E-commerce", "Customer Experience", "Supply Chain", "Merchandising"],
    "HR": ["Talent Management", "Organizational Design", "HR Analytics", "Change Management"],
    "Finance": ["Financial Planning", "Accounting", "Financial Reporting", "Budgeting"],
    "Marketing": ["Digital Marketing", "Brand Strategy", "Market Research", "Customer Analytics"],
    "IT": ["IT Infrastructure", "Software Development", "Cloud Services", "IT Security"]
}

# Common first and last names for diversity
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Andrew", "Emily", "Paul", "Donna", "Joshua", "Michelle",
    "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa", "Timothy", "Deborah",
    "Ronald", "Stephanie", "Jason", "Rebecca", "Edward", "Sharon", "Jeffrey", "Laura",
    "Ryan", "Cynthia", "Jacob", "Dorothy", "Gary", "Amy", "Nicholas", "Kathleen",
    "Eric", "Angela", "Jonathan", "Shirley", "Stephen", "Anna", "Larry", "Brenda",
    "Justin", "Pamela", "Scott", "Emma", "Brandon", "Nicole", "Benjamin", "Helen",
    "Samuel", "Samantha", "Raymond", "Katherine", "Gregory", "Christine", "Alexander", "Debra",
    "Frank", "Rachel", "Patrick", "Carolyn", "Raymond", "Janet", "Jack", "Catherine",
    "Dennis", "Maria", "Jerry", "Heather", "Tyler", "Diane", "Aaron", "Ruth"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
    "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
    "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers"
]

LOCATIONS = [
    "New York, NY", "San Francisco, CA", "Chicago, IL", "Boston, MA",
    "Los Angeles, CA", "Seattle, WA", "Washington, DC", "Atlanta, GA",
    "Dallas, TX", "Denver, CO", "Austin, TX", "Miami, FL"
]

CLIENT_NAMES = [
    "GlobalTech Industries", "Acme Manufacturing", "Premier Financial Group",
    "Healthcare Solutions Inc", "RetailCorp", "Energy Dynamics LLC",
    "First National Bank", "TechVentures", "Metropolitan Hospital System",
    "Continental Airlines", "AutoMakers United", "PharmaCo International",
    "Digital Media Group", "Telecommunications Plus", "Insurance Partners",
    "Real Estate Holdings", "Consumer Goods Corp", "Logistics Solutions",
    "Food & Beverage Co", "Chemicals International", "Aerospace Systems",
    "Construction Group", "Mining Resources", "Utilities Corporation",
    "Entertainment Studios", "Publishing House", "Education Services",
    "Hospitality Group", "Transportation Network", "Agriculture Products"
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Employee:
    employee_id: str
    first_name: str
    last_name: str
    email: str
    department: str
    job_title: str
    level: int
    hire_date: str
    termination_date: Optional[str]
    manager_id: Optional[str]
    base_salary: int
    location: str
    skills: str
    status: str  # active, terminated


@dataclass
class Project:
    project_id: str
    project_name: str
    client_name: str
    project_type: str
    start_date: str
    end_date: str
    budget: int
    status: str
    partner_id: str
    manager_id: str


@dataclass
class ProjectAssignment:
    assignment_id: str
    employee_id: str
    project_id: str
    start_date: str
    end_date: Optional[str]
    allocation_percentage: int
    role_on_project: str
    billable_rate: int


@dataclass
class TimeEntry:
    entry_id: str
    employee_id: str
    project_id: Optional[str]
    date: str
    hours: float
    billable_hours: float
    activity_type: str


@dataclass
class EmployeeEvent:
    event_id: str
    employee_id: str
    event_type: str
    event_date: str
    notes: str
    new_title: Optional[str]
    new_salary: Optional[int]


@dataclass
class TimeOff:
    time_off_id: str
    employee_id: str
    start_date: str
    end_date: str
    type: str
    status: str
    hours: float


@dataclass
class Payroll:
    payroll_id: str
    employee_id: str
    pay_period_start: str
    pay_period_end: str
    base_pay: float
    bonus: float
    deductions: float
    net_pay: float


# ============================================================================
# Generator Class
# ============================================================================

class ConsultingERPGenerator:
    def __init__(self, start_date: datetime, end_date: datetime,
                 start_employees: int, end_employees: int):
        self.start_date = start_date
        self.end_date = end_date
        self.start_employees = start_employees
        self.end_employees = end_employees
        self.simulation_days = (end_date - start_date).days

        # Storage
        self.employees: List[Employee] = []
        self.projects: List[Project] = []
        self.project_assignments: List[ProjectAssignment] = []
        self.time_entries: List[TimeEntry] = []
        self.employee_events: List[EmployeeEvent] = []
        self.time_off_records: List[TimeOff] = []
        self.payroll_records: List[Payroll] = []

        # Tracking
        self.used_emails = set()
        self.active_employees: Dict[str, Employee] = {}
        self.terminated_employees: Dict[str, Employee] = {}

        random.seed(42)  # For reproducibility

    def generate_unique_email(self, first_name: str, last_name: str) -> str:
        """Generate a unique email address."""
        base_email = f"{first_name.lower()}.{last_name.lower()}@consulting-firm.com"
        email = base_email
        counter = 1
        while email in self.used_emails:
            email = f"{first_name.lower()}.{last_name.lower()}{counter}@consulting-firm.com"
            counter += 1
        self.used_emails.add(email)
        return email

    def create_employee(self, hire_date: datetime, level: int = None) -> Employee:
        """Create a new employee with realistic attributes."""
        if level is None:
            # Weight by typical organizational structure
            level = random.choices(
                list(JOB_LEVELS.keys()),
                weights=[JOB_LEVELS[l]["weight"] for l in JOB_LEVELS.keys()]
            )[0]

        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        email = self.generate_unique_email(first_name, last_name)

        department = random.choice(DEPARTMENTS)
        job_title = random.choice(JOB_LEVELS[level]["titles"])
        salary_min, salary_max = JOB_LEVELS[level]["salary_range"]
        base_salary = random.randint(salary_min, salary_max)
        location = random.choice(LOCATIONS)

        # Assign 2-4 skills based on department
        dept_skills = SKILLS_BY_DEPT.get(department, ["General Consulting"])
        num_skills = random.randint(2, min(4, len(dept_skills)))
        skills = ", ".join(random.sample(dept_skills, num_skills))

        # Assign manager (someone from higher level)
        manager_id = None
        if level > 1 and self.active_employees:
            potential_managers = [
                emp for emp in self.active_employees.values()
                if emp.level >= level + 1 and emp.department == department
            ]
            if potential_managers:
                manager_id = random.choice(potential_managers).employee_id
            elif level <= 3:  # Junior employees can have managers from other departments
                potential_managers = [
                    emp for emp in self.active_employees.values()
                    if emp.level >= level + 1
                ]
                if potential_managers:
                    manager_id = random.choice(potential_managers).employee_id

        employee = Employee(
            employee_id=f"EMP{len(self.employees) + 1:05d}",
            first_name=first_name,
            last_name=last_name,
            email=email,
            department=department,
            job_title=job_title,
            level=level,
            hire_date=hire_date.strftime("%Y-%m-%d"),
            termination_date=None,
            manager_id=manager_id,
            base_salary=base_salary,
            location=location,
            skills=skills,
            status="active"
        )

        self.employees.append(employee)
        self.active_employees[employee.employee_id] = employee

        # Create hire event
        event = EmployeeEvent(
            event_id=f"EVT{len(self.employee_events) + 1:06d}",
            employee_id=employee.employee_id,
            event_type="hire",
            event_date=hire_date.strftime("%Y-%m-%d"),
            notes=f"Hired as {job_title} in {department}",
            new_title=job_title,
            new_salary=base_salary
        )
        self.employee_events.append(event)

        return employee

    def terminate_employee(self, employee: Employee, termination_date: datetime):
        """Terminate an employee."""
        employee.termination_date = termination_date.strftime("%Y-%m-%d")
        employee.status = "terminated"

        # Move to terminated list
        del self.active_employees[employee.employee_id]
        self.terminated_employees[employee.employee_id] = employee

        # Create termination event
        event = EmployeeEvent(
            event_id=f"EVT{len(self.employee_events) + 1:06d}",
            employee_id=employee.employee_id,
            event_type="termination",
            event_date=termination_date.strftime("%Y-%m-%d"),
            notes="Employee left the company",
            new_title=None,
            new_salary=None
        )
        self.employee_events.append(event)

        # End all active project assignments
        for assignment in self.project_assignments:
            if (assignment.employee_id == employee.employee_id and
                assignment.end_date is None):
                assignment.end_date = termination_date.strftime("%Y-%m-%d")

    def promote_employee(self, employee: Employee, promotion_date: datetime):
        """Promote an employee to the next level."""
        if employee.level >= 8:
            return  # Already at max level

        new_level = employee.level + 1
        new_title = random.choice(JOB_LEVELS[new_level]["titles"])
        salary_min, salary_max = JOB_LEVELS[new_level]["salary_range"]
        new_salary = random.randint(salary_min, salary_max)

        old_title = employee.job_title
        old_salary = employee.base_salary

        employee.level = new_level
        employee.job_title = new_title
        employee.base_salary = new_salary

        # Create promotion event
        event = EmployeeEvent(
            event_id=f"EVT{len(self.employee_events) + 1:06d}",
            employee_id=employee.employee_id,
            event_type="promotion",
            event_date=promotion_date.strftime("%Y-%m-%d"),
            notes=f"Promoted from {old_title} to {new_title}",
            new_title=new_title,
            new_salary=new_salary
        )
        self.employee_events.append(event)

    def create_project(self, start_date: datetime) -> Project:
        """Create a new consulting project."""
        # Project duration: 1-6 months (weighted towards 2-4 months)
        duration_days = random.choices(
            [30, 60, 90, 120, 150, 180],
            weights=[0.1, 0.25, 0.3, 0.2, 0.1, 0.05]
        )[0]
        end_date = start_date + timedelta(days=duration_days)

        # Ensure project doesn't extend beyond simulation period
        if end_date > self.end_date:
            end_date = self.end_date

        project_type = random.choice(PROJECT_TYPES)
        client_name = random.choice(CLIENT_NAMES)

        # Budget based on duration and team size (roughly $200-500 per hour, 40 hrs/week per person)
        weeks = duration_days / 7
        team_size = random.randint(3, 8)
        hourly_rate = random.randint(200, 500)
        budget = int(weeks * 40 * team_size * hourly_rate)

        # Assign partner and manager
        partners = [emp for emp in self.active_employees.values() if emp.level >= 6]
        managers = [emp for emp in self.active_employees.values() if emp.level >= 4]

        if not partners or not managers:
            return None

        partner = random.choice(partners)
        manager = random.choice(managers)

        project = Project(
            project_id=f"PRJ{len(self.projects) + 1:05d}",
            project_name=f"{project_type} - {client_name}",
            client_name=client_name,
            project_type=project_type,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            budget=budget,
            status="active",
            partner_id=partner.employee_id,
            manager_id=manager.employee_id
        )

        self.projects.append(project)
        return project

    def get_employee_max_allocation(self, employee_id: str, start_date: datetime, end_date: datetime) -> int:
        """Get the maximum allocation percentage for an employee during a date range."""
        date_allocations = defaultdict(int)

        for asn in self.project_assignments:
            if asn.employee_id != employee_id:
                continue

            asn_start = datetime.strptime(asn.start_date, "%Y-%m-%d")
            asn_end = datetime.strptime(asn.end_date, "%Y-%m-%d")

            # Check if assignments overlap
            if asn_start <= end_date and asn_end >= start_date:
                # Calculate overlapping period
                overlap_start = max(asn_start, start_date)
                overlap_end = min(asn_end, end_date)

                current = overlap_start
                while current <= overlap_end:
                    date_allocations[current.strftime("%Y-%m-%d")] += asn.allocation_percentage
                    current += timedelta(days=1)

        # Return maximum allocation across all days
        return max(date_allocations.values()) if date_allocations else 0

    def assign_employees_to_project(self, project: Project):
        """Assign employees to a project with realistic roles."""
        start_date = datetime.strptime(project.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(project.end_date, "%Y-%m-%d")

        # Partner - 10-20% allocation (working on multiple projects)
        partner_allocation = random.randint(10, 20)
        max_alloc = self.get_employee_max_allocation(project.partner_id, start_date, end_date)
        # Adjust allocation if would cause over-allocation, but always assign partner
        if max_alloc + partner_allocation > 98:  # Cap at 98% total for safety buffer
            partner_allocation = max(5, 98 - max_alloc)
        if partner_allocation >= 5:  # Always assign if at least 5% available
            self.create_assignment(project.partner_id, project, start_date, end_date,
                                  partner_allocation, "Partner")

        # Manager - 80-100% allocation
        manager_allocation = random.randint(80, 100)
        max_alloc = self.get_employee_max_allocation(project.manager_id, start_date, end_date)
        # Adjust allocation if would cause over-allocation
        if max_alloc + manager_allocation > 98:  # Cap at 98% total for safety buffer
            manager_allocation = max(10, 98 - max_alloc)
        if manager_allocation >= 10:  # Only assign if meaningful allocation
            self.create_assignment(project.manager_id, project, start_date, end_date,
                                  manager_allocation, "Project Manager")

        # Add 2-5 consultants/analysts
        num_team_members = random.randint(2, 5)
        junior_employees = [
            emp for emp in self.active_employees.values()
            if emp.level <= 3 and emp.employee_id not in [project.partner_id, project.manager_id]
        ]

        if junior_employees:
            # Try to find employees with capacity
            available_employees = []
            for emp in junior_employees:
                max_alloc = self.get_employee_max_allocation(emp.employee_id, start_date, end_date)
                if max_alloc < 80:  # Has at least 20% capacity
                    available_employees.append(emp)

            # If not enough available, use all junior employees
            if len(available_employees) < num_team_members:
                available_employees = junior_employees

            team_members = random.sample(available_employees, min(num_team_members, len(available_employees)))
            for emp in team_members:
                desired_allocation = random.randint(80, 100)
                max_alloc = self.get_employee_max_allocation(emp.employee_id, start_date, end_date)

                # Adjust allocation to avoid over-allocation
                if max_alloc + desired_allocation > 98:  # Cap at 98% total for safety buffer
                    desired_allocation = max(10, 98 - max_alloc)

                if desired_allocation >= 10:  # Only assign if meaningful
                    role = "Senior Consultant" if emp.level == 3 else "Consultant" if emp.level == 2 else "Analyst"
                    self.create_assignment(emp.employee_id, project, start_date, end_date,
                                          desired_allocation, role)

    def create_assignment(self, employee_id: str, project: Project,
                         start_date: datetime, end_date: datetime,
                         allocation: int, role: str):
        """Create a project assignment."""
        employee = self.active_employees.get(employee_id) or self.terminated_employees.get(employee_id)
        if not employee:
            return

        # Billable rate based on level (2-3x base salary / 2080 hours)
        hourly_base = employee.base_salary / 2080
        billable_rate = int(hourly_base * random.uniform(2.5, 3.5))

        assignment = ProjectAssignment(
            assignment_id=f"ASG{len(self.project_assignments) + 1:06d}",
            employee_id=employee_id,
            project_id=project.project_id,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            allocation_percentage=allocation,
            role_on_project=role,
            billable_rate=billable_rate
        )
        self.project_assignments.append(assignment)

    def generate_time_entries(self):
        """Generate daily time entries for all employees on their projects."""
        print("Generating time entries...")

        # Group assignments by employee
        assignments_by_employee: Dict[str, List[ProjectAssignment]] = {}
        for assignment in self.project_assignments:
            if assignment.employee_id not in assignments_by_employee:
                assignments_by_employee[assignment.employee_id] = []
            assignments_by_employee[assignment.employee_id].append(assignment)

        current_date = self.start_date
        while current_date <= self.end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                for employee_id, employee in self.active_employees.items():
                    # Find active assignments for this date
                    active_assignments = [
                        asn for asn in assignments_by_employee.get(employee_id, [])
                        if (datetime.strptime(asn.start_date, "%Y-%m-%d") <= current_date and
                            datetime.strptime(asn.end_date, "%Y-%m-%d") >= current_date)
                    ]

                    # Check if employee is on time off
                    on_time_off = any(
                        to for to in self.time_off_records
                        if (to.employee_id == employee_id and
                            datetime.strptime(to.start_date, "%Y-%m-%d") <= current_date and
                            datetime.strptime(to.end_date, "%Y-%m-%d") >= current_date)
                    )

                    if on_time_off:
                        continue

                    if active_assignments:
                        # Distribute 8 hours across active projects
                        total_allocation = sum(asn.allocation_percentage for asn in active_assignments)

                        for assignment in active_assignments:
                            allocation_ratio = assignment.allocation_percentage / total_allocation
                            hours = round(8 * allocation_ratio, 1)

                            # Some variation in hours (7-10 hours per day)
                            hours = max(0, hours + random.uniform(-1, 2))
                            billable_hours = hours * random.uniform(0.85, 1.0)  # 85-100% billable

                            entry = TimeEntry(
                                entry_id=f"TIME{len(self.time_entries) + 1:08d}",
                                employee_id=employee_id,
                                project_id=assignment.project_id,
                                date=current_date.strftime("%Y-%m-%d"),
                                hours=round(hours, 1),
                                billable_hours=round(billable_hours, 1),
                                activity_type="Project Work"
                            )
                            self.time_entries.append(entry)
                    else:
                        # "Beach time" - non-billable between projects
                        hours = random.uniform(4, 8)
                        entry = TimeEntry(
                            entry_id=f"TIME{len(self.time_entries) + 1:08d}",
                            employee_id=employee_id,
                            project_id=None,
                            date=current_date.strftime("%Y-%m-%d"),
                            hours=round(hours, 1),
                            billable_hours=0.0,
                            activity_type="Training" if random.random() < 0.3 else "Administrative"
                        )
                        self.time_entries.append(entry)

            current_date += timedelta(days=1)

    def generate_time_off(self):
        """Generate vacation and time off records."""
        print("Generating time off records...")

        for employee in self.employees:
            # Number of vacation days (15-25 days per year)
            days_per_year = random.randint(15, 25)
            vacation_days = int(days_per_year * (self.simulation_days / 365))

            # Generate 2-4 vacation periods
            num_vacations = random.randint(2, 4)

            for _ in range(num_vacations):
                # Random start date during employment
                hire_date = datetime.strptime(employee.hire_date, "%Y-%m-%d")
                term_date = datetime.strptime(employee.termination_date, "%Y-%m-%d") if employee.termination_date else self.end_date

                if (term_date - hire_date).days < 30:
                    continue

                days_employed = (min(term_date, self.end_date) - max(hire_date, self.start_date)).days
                if days_employed <= 0:
                    continue

                vacation_start = max(hire_date, self.start_date) + timedelta(days=random.randint(0, max(1, days_employed - 10)))
                vacation_days_this_period = random.randint(3, 10)
                vacation_end = vacation_start + timedelta(days=vacation_days_this_period)

                if vacation_end > min(term_date, self.end_date):
                    continue

                time_off = TimeOff(
                    time_off_id=f"TO{len(self.time_off_records) + 1:06d}",
                    employee_id=employee.employee_id,
                    start_date=vacation_start.strftime("%Y-%m-%d"),
                    end_date=vacation_end.strftime("%Y-%m-%d"),
                    type="Vacation",
                    status="Approved",
                    hours=vacation_days_this_period * 8
                )
                self.time_off_records.append(time_off)

            # Occasional sick days
            num_sick_days = random.randint(0, 5)
            for _ in range(num_sick_days):
                sick_date = max(hire_date, self.start_date) + timedelta(days=random.randint(0, max(1, days_employed)))
                if sick_date > min(term_date, self.end_date):
                    continue

                time_off = TimeOff(
                    time_off_id=f"TO{len(self.time_off_records) + 1:06d}",
                    employee_id=employee.employee_id,
                    start_date=sick_date.strftime("%Y-%m-%d"),
                    end_date=sick_date.strftime("%Y-%m-%d"),
                    type="Sick",
                    status="Approved",
                    hours=8
                )
                self.time_off_records.append(time_off)

    def generate_payroll(self):
        """Generate bi-weekly payroll records."""
        print("Generating payroll records...")

        current_date = self.start_date
        while current_date <= self.end_date:
            pay_period_end = current_date + timedelta(days=13)  # 2 weeks
            if pay_period_end > self.end_date:
                pay_period_end = self.end_date

            for employee in self.employees:
                hire_date = datetime.strptime(employee.hire_date, "%Y-%m-%d")
                term_date = datetime.strptime(employee.termination_date, "%Y-%m-%d") if employee.termination_date else None

                # Check if employee was active during this pay period
                if hire_date > pay_period_end:
                    continue
                if term_date and term_date < current_date:
                    continue

                # Calculate base pay for 2 weeks
                annual_salary = employee.base_salary
                bi_weekly_base = annual_salary / 26  # 26 pay periods per year

                # Random bonus (quarterly or annual)
                bonus = 0
                if random.random() < 0.08:  # ~8% chance (quarterly bonuses)
                    bonus = annual_salary * random.uniform(0.05, 0.15)

                # Deductions (taxes, benefits: roughly 25-30%)
                gross_pay = bi_weekly_base + bonus
                deduction_rate = random.uniform(0.25, 0.30)
                deductions = gross_pay * deduction_rate
                net_pay = gross_pay - deductions

                payroll = Payroll(
                    payroll_id=f"PAY{len(self.payroll_records) + 1:07d}",
                    employee_id=employee.employee_id,
                    pay_period_start=current_date.strftime("%Y-%m-%d"),
                    pay_period_end=pay_period_end.strftime("%Y-%m-%d"),
                    base_pay=round(bi_weekly_base, 2),
                    bonus=round(bonus, 2),
                    deductions=round(deductions, 2),
                    net_pay=round(net_pay, 2)
                )
                self.payroll_records.append(payroll)

            current_date = pay_period_end + timedelta(days=1)

    def run_simulation(self):
        """Run the complete simulation."""
        print(f"Starting ERP simulation from {self.start_date.date()} to {self.end_date.date()}")
        print(f"Target employees: {self.start_employees} → {self.end_employees}")
        print()

        # Phase 1: Create initial employee base
        print(f"Phase 1: Creating initial {self.start_employees} employees...")

        # Create senior leadership first (partners, principals)
        num_senior = int(self.start_employees * 0.05)  # 5% senior
        for i in range(num_senior):
            level = random.choice([6, 7, 8])
            hire_years_ago = random.randint(5, 15)
            hire_date = self.start_date - timedelta(days=hire_years_ago * 365)
            self.create_employee(hire_date, level=level)

        # Create middle management
        num_middle = int(self.start_employees * 0.15)  # 15% middle management
        for i in range(num_middle):
            level = random.choice([4, 5])
            hire_years_ago = random.randint(3, 10)
            hire_date = self.start_date - timedelta(days=hire_years_ago * 365)
            self.create_employee(hire_date, level=level)

        # Create rest of the workforce
        remaining = self.start_employees - len(self.employees)
        for i in range(remaining):
            hire_years_ago = random.randint(0, 8)
            hire_date = self.start_date - timedelta(days=hire_years_ago * 365)
            self.create_employee(hire_date)

        print(f"Created {len(self.employees)} initial employees")
        print()

        # Phase 2: Simulate employee lifecycle events
        print("Phase 2: Simulating employee lifecycle...")

        # Calculate how many net new employees we need
        net_change = self.end_employees - self.start_employees
        attrition_rate = 0.22  # 22% annual attrition

        # Calculate hires and terminations to achieve target with realistic attrition
        annual_terminations = int(self.start_employees * attrition_rate * (self.simulation_days / 365))
        annual_hires = annual_terminations + net_change

        # Distribute terminations throughout the period
        termination_dates = []
        for _ in range(annual_terminations):
            days_offset = random.randint(0, self.simulation_days)
            term_date = self.start_date + timedelta(days=days_offset)
            termination_dates.append(term_date)

        termination_dates.sort()

        # Terminate employees (favor junior employees)
        for term_date in termination_dates:
            if not self.active_employees:
                break

            # Weight towards junior employees (higher turnover)
            eligible_employees = [
                emp for emp in self.active_employees.values()
                if emp.level <= 4  # Junior to mid-level
            ]

            if not eligible_employees:
                eligible_employees = list(self.active_employees.values())

            if eligible_employees:
                employee = random.choice(eligible_employees)
                self.terminate_employee(employee, term_date)

        print(f"Simulated {len(termination_dates)} terminations")

        # Distribute hires throughout the period
        hire_dates = []
        for _ in range(annual_hires):
            days_offset = random.randint(0, self.simulation_days)
            hire_date = self.start_date + timedelta(days=days_offset)
            hire_dates.append(hire_date)

        hire_dates.sort()

        for hire_date in hire_dates:
            self.create_employee(hire_date)

        print(f"Simulated {len(hire_dates)} new hires")

        # Promotions (roughly 15% of employees per year)
        num_promotions = int(len(self.employees) * 0.15 * (self.simulation_days / 365))
        for _ in range(num_promotions):
            eligible = [emp for emp in self.active_employees.values() if emp.level < 8]
            if eligible:
                employee = random.choice(eligible)
                promo_date = self.start_date + timedelta(days=random.randint(0, self.simulation_days))
                self.promote_employee(employee, promo_date)

        print(f"Simulated {num_promotions} promotions")
        print(f"Final employee count: {len(self.active_employees)} active")
        print()

        # Phase 3: Create projects
        print("Phase 3: Creating projects...")

        # Average 1 project per 6 employees, with some running concurrently
        num_projects = int(self.start_employees / 6 * (self.simulation_days / 120))  # 4-month avg project

        for i in range(num_projects):
            project_start = self.start_date + timedelta(days=random.randint(0, max(1, self.simulation_days - 60)))
            project = self.create_project(project_start)
            if project:
                self.assign_employees_to_project(project)

            if (i + 1) % 20 == 0:
                print(f"Created {i + 1} projects...")

        print(f"Created {len(self.projects)} projects")
        print()

        # Phase 4: Generate time off (before time entries!)
        self.generate_time_off()
        print(f"Generated {len(self.time_off_records)} time off records")
        print()

        # Phase 5: Generate time entries
        self.generate_time_entries()
        print(f"Generated {len(self.time_entries)} time entries")
        print()

        # Phase 6: Generate payroll
        self.generate_payroll()
        print(f"Generated {len(self.payroll_records)} payroll records")
        print()

    def export_to_csv(self, output_dir: str = "."):
        """Export all data to CSV files."""
        print("Exporting data to CSV files...")

        # Employees
        with open(f"{output_dir}/employees.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=Employee.__annotations__.keys())
            writer.writeheader()
            for emp in self.employees:
                writer.writerow(asdict(emp))
        print(f"✓ Exported employees.csv ({len(self.employees)} records)")

        # Projects
        with open(f"{output_dir}/projects.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=Project.__annotations__.keys())
            writer.writeheader()
            for proj in self.projects:
                writer.writerow(asdict(proj))
        print(f"✓ Exported projects.csv ({len(self.projects)} records)")

        # Project Assignments
        with open(f"{output_dir}/project_assignments.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ProjectAssignment.__annotations__.keys())
            writer.writeheader()
            for asn in self.project_assignments:
                writer.writerow(asdict(asn))
        print(f"✓ Exported project_assignments.csv ({len(self.project_assignments)} records)")

        # Time Entries
        with open(f"{output_dir}/time_entries.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=TimeEntry.__annotations__.keys())
            writer.writeheader()
            for entry in self.time_entries:
                writer.writerow(asdict(entry))
        print(f"✓ Exported time_entries.csv ({len(self.time_entries)} records)")

        # Employee Events
        with open(f"{output_dir}/employee_events.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=EmployeeEvent.__annotations__.keys())
            writer.writeheader()
            for event in self.employee_events:
                writer.writerow(asdict(event))
        print(f"✓ Exported employee_events.csv ({len(self.employee_events)} records)")

        # Time Off
        with open(f"{output_dir}/time_off.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=TimeOff.__annotations__.keys())
            writer.writeheader()
            for to in self.time_off_records:
                writer.writerow(asdict(to))
        print(f"✓ Exported time_off.csv ({len(self.time_off_records)} records)")

        # Payroll
        with open(f"{output_dir}/payroll.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=Payroll.__annotations__.keys())
            writer.writeheader()
            for pay in self.payroll_records:
                writer.writerow(asdict(pay))
        print(f"✓ Exported payroll.csv ({len(self.payroll_records)} records)")

        print("\n✓ All CSV files exported successfully!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate realistic consulting business ERP data'
    )
    parser.add_argument('--start_date', type=str, required=True,
                       help='Simulation start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True,
                       help='Simulation end date (YYYY-MM-DD)')
    parser.add_argument('--start_employees', type=int, required=True,
                       help='Number of employees at start')
    parser.add_argument('--end_employees', type=int, required=True,
                       help='Number of employees at end')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for CSV files (default: current directory)')

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Validate inputs
    if end_date <= start_date:
        print("Error: end_date must be after start_date")
        return

    if args.start_employees <= 0 or args.end_employees <= 0:
        print("Error: employee counts must be positive")
        return

    # Create generator and run simulation
    generator = ConsultingERPGenerator(
        start_date=start_date,
        end_date=end_date,
        start_employees=args.start_employees,
        end_employees=args.end_employees
    )

    generator.run_simulation()
    generator.export_to_csv(args.output_dir)

    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

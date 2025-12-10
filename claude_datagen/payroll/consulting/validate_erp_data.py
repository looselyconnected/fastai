"""
ERP Data Validation Script

Validates the integrity and business logic of generated consulting ERP data.
Checks invariants, consistency, and realistic business constraints.

Usage:
    python validate_erp_data.py [--data_dir .]
"""

import argparse
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import sys


class ERPDataValidator:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.errors = []
        self.warnings = []

        # Data storage
        self.employees = []
        self.projects = []
        self.assignments = []
        self.time_entries = []
        self.employee_events = []
        self.time_off = []
        self.payroll = []

        # Indices for fast lookup
        self.employees_by_id = {}
        self.projects_by_id = {}
        self.assignments_by_employee = defaultdict(list)
        self.assignments_by_project = defaultdict(list)
        self.time_entries_by_employee = defaultdict(list)
        self.time_entries_by_date = defaultdict(list)
        self.time_off_by_employee = defaultdict(list)
        self.events_by_employee = defaultdict(list)

    def load_data(self):
        """Load all CSV files."""
        print("Loading data files...")

        # Load employees
        with open(f"{self.data_dir}/employees.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.employees = list(reader)
            self.employees_by_id = {emp['employee_id']: emp for emp in self.employees}
        print(f"✓ Loaded {len(self.employees)} employees")

        # Load projects
        with open(f"{self.data_dir}/projects.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.projects = list(reader)
            self.projects_by_id = {proj['project_id']: proj for proj in self.projects}
        print(f"✓ Loaded {len(self.projects)} projects")

        # Load assignments
        with open(f"{self.data_dir}/project_assignments.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.assignments = list(reader)
            for asn in self.assignments:
                self.assignments_by_employee[asn['employee_id']].append(asn)
                self.assignments_by_project[asn['project_id']].append(asn)
        print(f"✓ Loaded {len(self.assignments)} project assignments")

        # Load time entries
        with open(f"{self.data_dir}/time_entries.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.time_entries = list(reader)
            for entry in self.time_entries:
                self.time_entries_by_employee[entry['employee_id']].append(entry)
                self.time_entries_by_date[entry['date']].append(entry)
        print(f"✓ Loaded {len(self.time_entries)} time entries")

        # Load employee events
        with open(f"{self.data_dir}/employee_events.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.employee_events = list(reader)
            for event in self.employee_events:
                self.events_by_employee[event['employee_id']].append(event)
        print(f"✓ Loaded {len(self.employee_events)} employee events")

        # Load time off
        with open(f"{self.data_dir}/time_off.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.time_off = list(reader)
            for to in self.time_off:
                self.time_off_by_employee[to['employee_id']].append(to)
        print(f"✓ Loaded {len(self.time_off)} time off records")

        # Load payroll
        with open(f"{self.data_dir}/payroll.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.payroll = list(reader)
        print(f"✓ Loaded {len(self.payroll)} payroll records")

        print()

    def add_error(self, category: str, message: str):
        """Add an error."""
        self.errors.append(f"[{category}] {message}")

    def add_warning(self, category: str, message: str):
        """Add a warning."""
        self.warnings.append(f"[{category}] {message}")

    def validate_employee_data(self):
        """Validate employee data integrity."""
        print("Validating employee data integrity...")

        seen_emails = set()
        salary_ranges = {
            '1': (80000, 95000), '2': (135000, 165000), '3': (165000, 195000),
            '4': (180000, 230000), '5': (220000, 280000), '6': (280000, 400000),
            '7': (400000, 650000), '8': (600000, 1000000)
        }

        for emp in self.employees:
            emp_id = emp['employee_id']

            # Check unique emails
            if emp['email'] in seen_emails:
                self.add_error("EMPLOYEE", f"{emp_id}: Duplicate email {emp['email']}")
            seen_emails.add(emp['email'])

            # Check hire/termination date logic
            if emp['hire_date']:
                hire_date = datetime.strptime(emp['hire_date'], "%Y-%m-%d")
                if emp['termination_date']:
                    term_date = datetime.strptime(emp['termination_date'], "%Y-%m-%d")
                    if term_date <= hire_date:
                        self.add_error("EMPLOYEE", f"{emp_id}: Termination date {emp['termination_date']} is before or same as hire date {emp['hire_date']}")

            # Check salary range
            level = emp['level']
            salary = int(emp['base_salary'])
            if level in salary_ranges:
                min_sal, max_sal = salary_ranges[level]
                if salary < min_sal or salary > max_sal:
                    self.add_warning("EMPLOYEE", f"{emp_id}: Salary ${salary:,} outside expected range ${min_sal:,}-${max_sal:,} for level {level}")

            # Check manager relationship
            if emp['manager_id']:
                manager = self.employees_by_id.get(emp['manager_id'])
                if not manager:
                    self.add_error("EMPLOYEE", f"{emp_id}: Manager {emp['manager_id']} does not exist")
                elif int(manager['level']) < int(emp['level']):
                    self.add_warning("EMPLOYEE", f"{emp_id}: Manager {emp['manager_id']} has lower level ({manager['level']}) than employee ({emp['level']})")

        print(f"✓ Employee data validation complete")

    def validate_project_staffing(self):
        """Validate project staffing - right number of people with right roles."""
        print("Validating project staffing...")

        for project in self.projects:
            proj_id = project['project_id']
            assignments = self.assignments_by_project[proj_id]

            if not assignments:
                self.add_error("PROJECT", f"{proj_id}: No employees assigned to project")
                continue

            # Check partner exists
            partner_id = project['partner_id']
            partner_assigned = any(a['employee_id'] == partner_id for a in assignments)
            if not partner_assigned:
                self.add_error("PROJECT", f"{proj_id}: Partner {partner_id} not assigned to project")

            # Check manager exists
            manager_id = project['manager_id']
            manager_assigned = any(a['employee_id'] == manager_id for a in assignments)
            if not manager_assigned:
                self.add_error("PROJECT", f"{proj_id}: Manager {manager_id} not assigned to project")

            # Check team size (typical 3-8 people)
            team_size = len(assignments)
            if team_size < 2:
                self.add_warning("PROJECT", f"{proj_id}: Very small team size ({team_size})")
            elif team_size > 10:
                self.add_warning("PROJECT", f"{proj_id}: Very large team size ({team_size})")

            # Check role distribution
            roles = [a['role_on_project'] for a in assignments]
            has_partner = any('Partner' in r for r in roles)
            has_manager = any('Manager' in r for r in roles)

            if not has_partner:
                self.add_warning("PROJECT", f"{proj_id}: No partner role assigned")
            if not has_manager:
                self.add_warning("PROJECT", f"{proj_id}: No manager role assigned")

            # Check that assigned employees exist and have appropriate levels
            for asn in assignments:
                emp = self.employees_by_id.get(asn['employee_id'])
                if not emp:
                    self.add_error("PROJECT", f"{proj_id}: Assigned employee {asn['employee_id']} does not exist")
                    continue

                # Partners should be level 6+
                if 'Partner' in asn['role_on_project'] and int(emp['level']) < 6:
                    self.add_warning("PROJECT", f"{proj_id}: Employee {asn['employee_id']} has partner role but is level {emp['level']}")

                # Managers should be level 4+
                if 'Manager' in asn['role_on_project'] and int(emp['level']) < 4:
                    self.add_warning("PROJECT", f"{proj_id}: Employee {asn['employee_id']} has manager role but is level {emp['level']}")

        print(f"✓ Project staffing validation complete")

    def validate_employee_allocation(self):
        """Validate that no employee is over-allocated at any point in time."""
        print("Validating employee allocations (checking for over-allocation)...")

        over_allocation_count = 0

        for emp_id, assignments in self.assignments_by_employee.items():
            if not assignments:
                continue

            # Get all date ranges for this employee
            date_allocations = defaultdict(int)

            for asn in assignments:
                start_date = datetime.strptime(asn['start_date'], "%Y-%m-%d")
                end_date = datetime.strptime(asn['end_date'], "%Y-%m-%d")
                allocation = int(asn['allocation_percentage'])

                # For each day in the assignment, add to allocation
                current = start_date
                while current <= end_date:
                    date_allocations[current.strftime("%Y-%m-%d")] += allocation
                    current += timedelta(days=1)

            # Check for over-allocation
            for date, total_allocation in date_allocations.items():
                if total_allocation > 100:
                    self.add_error("ALLOCATION",
                                 f"Employee {emp_id} over-allocated on {date}: {total_allocation}%")
                    over_allocation_count += 1
                    # Only report first few instances per employee to avoid spam
                    if over_allocation_count > 5:
                        break

            if over_allocation_count > 5:
                break

        if over_allocation_count == 0:
            print(f"✓ No over-allocations found")
        else:
            print(f"✓ Employee allocation validation complete")

    def validate_assignment_consistency(self):
        """Validate project assignment consistency."""
        print("Validating assignment consistency...")

        for asn in self.assignments:
            asn_id = asn['assignment_id']

            # Check employee exists
            if asn['employee_id'] not in self.employees_by_id:
                self.add_error("ASSIGNMENT", f"{asn_id}: Employee {asn['employee_id']} does not exist")

            # Check project exists
            if asn['project_id'] not in self.projects_by_id:
                self.add_error("ASSIGNMENT", f"{asn_id}: Project {asn['project_id']} does not exist")
                continue

            # Check date consistency
            asn_start = datetime.strptime(asn['start_date'], "%Y-%m-%d")
            asn_end = datetime.strptime(asn['end_date'], "%Y-%m-%d")

            if asn_end < asn_start:
                self.add_error("ASSIGNMENT", f"{asn_id}: End date {asn['end_date']} before start date {asn['start_date']}")

            # Check assignment dates are within project dates
            project = self.projects_by_id[asn['project_id']]
            proj_start = datetime.strptime(project['start_date'], "%Y-%m-%d")
            proj_end = datetime.strptime(project['end_date'], "%Y-%m-%d")

            if asn_start < proj_start:
                self.add_warning("ASSIGNMENT", f"{asn_id}: Assignment starts before project start")
            if asn_end > proj_end:
                self.add_warning("ASSIGNMENT", f"{asn_id}: Assignment ends after project end")

            # Check allocation percentage
            allocation = int(asn['allocation_percentage'])
            if allocation < 0 or allocation > 100:
                self.add_error("ASSIGNMENT", f"{asn_id}: Invalid allocation {allocation}%")

            # Check billable rate
            billable_rate = int(asn['billable_rate'])
            if billable_rate <= 0:
                self.add_error("ASSIGNMENT", f"{asn_id}: Invalid billable rate ${billable_rate}")

        print(f"✓ Assignment consistency validation complete")

    def validate_time_entries(self):
        """Validate time entry consistency."""
        print("Validating time entries...")

        # Sample validation (checking first 1000 entries to keep it fast)
        sample_size = min(1000, len(self.time_entries))

        for i, entry in enumerate(self.time_entries[:sample_size]):
            entry_id = entry['entry_id']

            # Check employee exists
            if entry['employee_id'] not in self.employees_by_id:
                self.add_error("TIME_ENTRY", f"{entry_id}: Employee {entry['employee_id']} does not exist")
                continue

            # Check hours are reasonable
            hours = float(entry['hours'])
            billable_hours = float(entry['billable_hours'])

            if hours < 0:
                self.add_error("TIME_ENTRY", f"{entry_id}: Negative hours {hours}")
            if hours > 16:
                self.add_warning("TIME_ENTRY", f"{entry_id}: Unusually high hours {hours}")

            if billable_hours < 0:
                self.add_error("TIME_ENTRY", f"{entry_id}: Negative billable hours {billable_hours}")
            if billable_hours > hours:
                self.add_error("TIME_ENTRY", f"{entry_id}: Billable hours {billable_hours} exceed total hours {hours}")

            # Check no entries on weekends
            entry_date = datetime.strptime(entry['date'], "%Y-%m-%d")
            if entry_date.weekday() >= 5:
                self.add_error("TIME_ENTRY", f"{entry_id}: Time entry on weekend {entry['date']}")

            # If project_id is specified, check assignment exists
            if entry['project_id']:
                emp_assignments = self.assignments_by_employee[entry['employee_id']]
                has_assignment = any(
                    a['project_id'] == entry['project_id'] and
                    datetime.strptime(a['start_date'], "%Y-%m-%d") <= entry_date <=
                    datetime.strptime(a['end_date'], "%Y-%m-%d")
                    for a in emp_assignments
                )
                if not has_assignment:
                    self.add_warning("TIME_ENTRY", f"{entry_id}: Time entry for project without assignment")

        print(f"✓ Time entry validation complete (sampled {sample_size} entries)")

    def validate_time_off_conflicts(self):
        """Validate no time entries during time off."""
        print("Validating time off conflicts...")

        conflicts = 0

        for emp_id, time_off_records in self.time_off_by_employee.items():
            time_entries = self.time_entries_by_employee.get(emp_id, [])

            for to in time_off_records:
                to_start = datetime.strptime(to['start_date'], "%Y-%m-%d")
                to_end = datetime.strptime(to['end_date'], "%Y-%m-%d")

                # Check for time entries during time off
                for entry in time_entries:
                    entry_date = datetime.strptime(entry['date'], "%Y-%m-%d")
                    if to_start <= entry_date <= to_end:
                        self.add_error("TIME_OFF",
                                     f"Employee {emp_id} has time entry on {entry['date']} during time off period {to['start_date']} to {to['end_date']}")
                        conflicts += 1
                        if conflicts > 10:  # Limit output
                            break

                if conflicts > 10:
                    break

            if conflicts > 10:
                break

        if conflicts == 0:
            print(f"✓ No time off conflicts found")
        else:
            print(f"✓ Time off validation complete")

    def validate_payroll(self):
        """Validate payroll calculations."""
        print("Validating payroll...")

        # Sample validation
        sample_size = min(100, len(self.payroll))

        for pay in self.payroll[:sample_size]:
            pay_id = pay['payroll_id']

            # Check employee exists
            if pay['employee_id'] not in self.employees_by_id:
                self.add_error("PAYROLL", f"{pay_id}: Employee {pay['employee_id']} does not exist")
                continue

            # Check date logic
            start = datetime.strptime(pay['pay_period_start'], "%Y-%m-%d")
            end = datetime.strptime(pay['pay_period_end'], "%Y-%m-%d")

            if end < start:
                self.add_error("PAYROLL", f"{pay_id}: End date before start date")

            # Check calculation
            base_pay = float(pay['base_pay'])
            bonus = float(pay['bonus'])
            deductions = float(pay['deductions'])
            net_pay = float(pay['net_pay'])

            expected_net = base_pay + bonus - deductions
            if abs(expected_net - net_pay) > 0.02:  # Allow small rounding errors
                self.add_error("PAYROLL", f"{pay_id}: Net pay calculation incorrect. Expected {expected_net:.2f}, got {net_pay:.2f}")

            # Check values are reasonable
            if base_pay < 0:
                self.add_error("PAYROLL", f"{pay_id}: Negative base pay")
            if bonus < 0:
                self.add_error("PAYROLL", f"{pay_id}: Negative bonus")
            if deductions < 0:
                self.add_error("PAYROLL", f"{pay_id}: Negative deductions")

        print(f"✓ Payroll validation complete (sampled {sample_size} records)")

    def validate_employee_events(self):
        """Validate employee events match employee records."""
        print("Validating employee events...")

        for emp_id, events in self.events_by_employee.items():
            emp = self.employees_by_id.get(emp_id)
            if not emp:
                self.add_error("EVENT", f"Events exist for non-existent employee {emp_id}")
                continue

            # Check hire event
            hire_events = [e for e in events if e['event_type'] == 'hire']
            if hire_events:
                hire_event = hire_events[0]
                if hire_event['event_date'] != emp['hire_date']:
                    self.add_warning("EVENT", f"{emp_id}: Hire event date {hire_event['event_date']} doesn't match hire date {emp['hire_date']}")

            # Check termination event
            term_events = [e for e in events if e['event_type'] == 'termination']
            if term_events and emp['termination_date']:
                term_event = term_events[0]
                if term_event['event_date'] != emp['termination_date']:
                    self.add_warning("EVENT", f"{emp_id}: Termination event date doesn't match termination date")

        print(f"✓ Employee events validation complete")

    def calculate_statistics(self):
        """Calculate and display summary statistics."""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)

        # Employee statistics
        active_employees = [e for e in self.employees if e['status'] == 'active']
        terminated_employees = [e for e in self.employees if e['status'] == 'terminated']

        print(f"\nEmployees:")
        print(f"  Total: {len(self.employees)}")
        print(f"  Active: {len(active_employees)}")
        print(f"  Terminated: {len(terminated_employees)}")

        # Level distribution
        level_dist = defaultdict(int)
        for emp in active_employees:
            level_dist[emp['level']] += 1

        print(f"\n  Active by Level:")
        for level in sorted(level_dist.keys()):
            count = level_dist[level]
            pct = count / len(active_employees) * 100
            print(f"    Level {level}: {count} ({pct:.1f}%)")

        # Project statistics
        print(f"\nProjects:")
        print(f"  Total: {len(self.projects)}")

        team_sizes = [len(self.assignments_by_project[p['project_id']]) for p in self.projects]
        if team_sizes:
            print(f"  Average team size: {sum(team_sizes)/len(team_sizes):.1f}")
            print(f"  Min team size: {min(team_sizes)}")
            print(f"  Max team size: {max(team_sizes)}")

        # Assignment statistics
        print(f"\nAssignments:")
        print(f"  Total: {len(self.assignments)}")

        allocations = [int(a['allocation_percentage']) for a in self.assignments]
        if allocations:
            print(f"  Average allocation: {sum(allocations)/len(allocations):.1f}%")

        # Time entry statistics
        total_hours = sum(float(e['hours']) for e in self.time_entries)
        total_billable = sum(float(e['billable_hours']) for e in self.time_entries)

        print(f"\nTime Entries:")
        print(f"  Total: {len(self.time_entries):,}")
        print(f"  Total hours: {total_hours:,.1f}")
        print(f"  Billable hours: {total_billable:,.1f}")
        print(f"  Utilization rate: {total_billable/total_hours*100:.1f}%")

        # Payroll statistics
        total_payroll = sum(float(p['net_pay']) for p in self.payroll)

        print(f"\nPayroll:")
        print(f"  Total: {len(self.payroll):,} records")
        print(f"  Total net pay: ${total_payroll:,.2f}")

        # Time off statistics
        print(f"\nTime Off:")
        print(f"  Total records: {len(self.time_off)}")
        vacation = len([t for t in self.time_off if t['type'] == 'Vacation'])
        sick = len([t for t in self.time_off if t['type'] == 'Sick'])
        print(f"  Vacation: {vacation}")
        print(f"  Sick: {sick}")

    def run_all_validations(self):
        """Run all validation checks."""
        print("="*70)
        print("STARTING ERP DATA VALIDATION")
        print("="*70)
        print()

        self.load_data()

        print("="*70)
        print("RUNNING VALIDATIONS")
        print("="*70)
        print()

        self.validate_employee_data()
        self.validate_project_staffing()
        self.validate_employee_allocation()
        self.validate_assignment_consistency()
        self.validate_time_entries()
        self.validate_time_off_conflicts()
        self.validate_payroll()
        self.validate_employee_events()

        self.calculate_statistics()

        # Print results
        print("\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)

        if not self.errors and not self.warnings:
            print("\n✅ ALL VALIDATIONS PASSED! No errors or warnings found.")
        else:
            if self.errors:
                print(f"\n❌ ERRORS: {len(self.errors)}")
                print("-"*70)
                for error in self.errors[:20]:  # Show first 20
                    print(f"  {error}")
                if len(self.errors) > 20:
                    print(f"  ... and {len(self.errors) - 20} more errors")

            if self.warnings:
                print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
                print("-"*70)
                for warning in self.warnings[:20]:  # Show first 20
                    print(f"  {warning}")
                if len(self.warnings) > 20:
                    print(f"  ... and {len(self.warnings) - 20} more warnings")

        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)

        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Validate consulting ERP data'
    )
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory containing CSV files (default: current directory)')

    args = parser.parse_args()

    validator = ERPDataValidator(args.data_dir)
    success = validator.run_all_validations()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

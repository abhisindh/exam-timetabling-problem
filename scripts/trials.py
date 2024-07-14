import random
import math
import numpy as np


def euclidean_distance(a, b, var='both'):
    """
    Calculate the Euclidean distance between two timetables.
    
    Parameters:
    a (Timetable): The first timetable.
    b (Timetable): The second timetable.
    var (str): The variable to consider for distance calculation ('both', 'room', 'date').
    
    Returns:
    float: The Euclidean distance between timetables a and b.
    """
    if var == 'both':
        return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a.r_a + a.d_a, b.r_a + b.d_a)))
    elif var == 'room':
        return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a.r_a, b.r_a)))
    elif var == 'date':
        return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a.d_a, b.d_a)))

def calculate_proximity_penalty(timetable, students_courses, days_assigned):
    """
    Calculate the proximity penalty for a given timetable.
    
    Parameters:
    timetable (Timetable): The timetable.
    students_courses (list): A list of lists where each sublist contains the courses a student is taking.
    days_assigned (dict): A dictionary mapping each course to the day it is assigned.
    
    Returns:
    int: The total proximity penalty for the timetable.
    """
    total_penalty = 0
    for student_courses in students_courses:
        for i in range(len(student_courses)):
            for j in range(i + 1, len(student_courses)):
                course1 = student_courses[i]
                course2 = student_courses[j]
                day1 = days_assigned.get(course1, -1)
                day2 = days_assigned.get(course2, -1)
                if day1 != -1 and day2 != -1:
                    proximity = abs(day1 - day2)
                    penalty = proximity ** 2
                    total_penalty += penalty
    return total_penalty

def create_timetable(courses, timeslots, rooms, students_courses):
    """
    Create a timetable by assigning timeslots and rooms to courses.
    
    Parameters:
    courses (list): The list of courses.
    timeslots (list): The list of available timeslots.
    rooms (list): The list of available rooms.
    students_courses (list): A list of lists where each sublist contains the courses a student is taking.
    
    Returns:
    dict, dict: The timetable and the assigned days for each course.
    """
    timetable = {}
    days_assigned = {}
    for course in courses:
        assigned = False
        while not assigned:
            timeslot = random.choice(timeslots)
            for room in rooms:
                if all(timeslot not in timetable.get((c, room), []) for c in courses):
                    timetable[(course, room)] = timeslot
                    days_assigned[course] = timeslot[0]
                    assigned = True
                    break
            if not assigned:
                timeslot = random.choice(timeslots)
    return timetable, days_assigned

def repair_timetable(timetable, rooms, max_iterations=1000):
    """
    Repair a timetable by resolving room conflicts.
    
    Parameters:
    timetable (dict): The timetable to be repaired.
    rooms (list): The list of available rooms.
    max_iterations (int): The maximum number of iterations for the repair process.
    
    Returns:
    dict: The repaired timetable.
    """
    iterations = 0
    while iterations < max_iterations:
        violations = find_violations(timetable, rooms)
        if not violations:
            break
        for violation in violations:
            course, conflicting_course, room = violation
            new_room = random.choice([r for r in rooms if r != room])
            timetable[(course, new_room)] = timetable.pop((course, room))
        iterations += 1
    return timetable

def find_violations(timetable, rooms):
    """
    Find violations in the timetable where multiple courses are assigned to the same room at the same timeslot.
    
    Parameters:
    timetable (dict): The timetable to be checked.
    rooms (list): The list of available rooms.
    
    Returns:
    list: A list of violations found in the timetable.
    """
    violations = []
    for (course, room), timeslot in timetable.items():
        for other_course in timetable:
            if other_course != course and timetable.get((other_course, room)) == timeslot:
                violations.append((course, other_course, room))
    return violations

def normalize_violations(violations, max_violation):
    """
    Normalize the violations based on the maximum possible violation.
    
    Parameters:
    violations (list): The list of violations.
    max_violation (int): The maximum possible violation.
    
    Returns:
    list: The normalized violations.
    """
    return [violation / max_violation for violation in violations]

def aggregate_normalized_violations(normalized_violations):
    """
    Aggregate the normalized violations to a single score.
    
    Parameters:
    normalized_violations (list): The list of normalized violations.
    
    Returns:
    float: The aggregated score.
    """
    return sum(normalized_violations)

def generate_initial_population(courses, timeslots, rooms, students_courses, population_size=50):
    """
    Generate the initial population for the genetic algorithm.
    
    Parameters:
    courses (list): The list of courses.
    timeslots (list): The list of available timeslots.
    rooms (list): The list of available rooms.
    students_courses (list): A list of lists where each sublist contains the courses a student is taking.
    population_size (int): The size of the population.
    
    Returns:
    list: The initial population.
    """
    population = []
    for _ in range(population_size):
        timetable, days_assigned = create_timetable(courses, timeslots, rooms, students_courses)
        population.append((timetable, days_assigned))
    return population

def genetic_algorithm(courses, timeslots, rooms, students_courses, generations=100, population_size=50):
    """
    Run the genetic algorithm to find the best timetable.
    
    Parameters:
    courses (list): The list of courses.
    timeslots (list): The list of available timeslots.
    rooms (list): The list of available rooms.
    students_courses (list): A list of lists where each sublist contains the courses a student is taking.
    generations (int): The number of generations.
    population_size (int): The size of the population.
    
    Returns:
    tuple: The best timetable and its proximity penalty.
    """
    population = generate_initial_population(courses, timeslots, rooms, students_courses, population_size)
    for generation in range(generations):
        new_population = []
        for timetable, days_assigned in population:
            repaired_timetable = repair_timetable(timetable, rooms)
            proximity_penalty = calculate_proximity_penalty(repaired_timetable, students_courses, days_assigned)
            new_population.append((repaired_timetable, proximity_penalty))
        population = sorted(new_population, key=lambda x: x[1])
        population = population[:population_size]
    return population[0]

def find_donor_solution(timetable, donor_timetables):
    """
    Find a donor solution from the population that does not conflict with the current timetable.
    
    Parameters:
    timetable (dict): The current timetable.
    donor_timetables (list): The list of donor timetables.
    
    Returns:
    dict: The selected donor timetable.
    """
    for donor_timetable in donor_timetables:
        if all(donor_timetable.get(course) != timetable.get(course) for course in timetable):
            return donor_timetable
    return random.choice(donor_timetables)

def repair_infeasible_solution(timetable, donor_timetables):
    """
    Repair an infeasible solution by resolving conflicts using donor timetables.
    
    Parameters:
    timetable (dict): The timetable to be repaired.
    donor_timetables (list): The list of donor timetables.
    
    Returns:
    dict: The repaired timetable.
    """
    violations = find_violations(timetable, rooms)
    for violation in violations:
        course, conflicting_course, room = violation
        donor_timetable = find_donor_solution(timetable, donor_timetables)
        timetable[(course, room)] = donor_timetable[course]
    return timetable

def repair_function(population):
    """
    Repair the population of timetables by resolving conflicts using donor solutions.
    
    Parameters:
    population (list): The population of timetables.
    
    Returns:
    list: The repaired population.
    """
    repaired_population = []
    for timetable, proximity_penalty in population:
        repaired_timetable = repair_infeasible_solution(timetable, [t for t, _ in population])
        repaired_proximity_penalty = calculate_proximity_penalty(repaired_timetable, students_courses, days_assigned)
        repaired_population.append((repaired_timetable, repaired_proximity_penalty))
    return repaired_population

def main():
    # Define the courses, timeslots, rooms, and students_courses
    courses = ['MA101', 'PY101', 'CH101', 'CS101']
    timeslots = [(day, slot) for day in range(5) for slot in range(4)]
    rooms = ['R1', 'R2', 'R3']
    students_courses = [['MA101', 'PY101'], ['CH101', 'CS101'], ['MA101', 'CS101'], ['PY101', 'CH101']]

    # Run the genetic algorithm to find the best timetable
    best_timetable, best_penalty = genetic_algorithm(courses, timeslots, rooms, students_courses)

    # Print the best timetable and its proximity penalty
    print("Best Timetable:", best_timetable)
    print("Best Proximity Penalty:", best_penalty)

if __name__ == "__main__":
    main()

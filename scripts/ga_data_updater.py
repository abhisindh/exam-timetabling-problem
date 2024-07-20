from utils import *
import csv
import time
import numpy as np

# Assuming Instance and other necessary functions/classes are imported or defined elsewhere

instance_list = ['car-s-91','car-f-92','ear-f-83', 'kfu-s-93', 'sta-f-83', 'tre-s-92', 'uta-s-92', 'yor-f-83']
instance_list = ['hec-s-92', 'lse-f-91', 'rye-s-93', 'ute-s-92']

# Initialize CSV file
csv_file = "../docs/performance_tracking/notebook.csv"

# Open the CSV file in append mode
with open(csv_file, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["File Name", "Trial Number", "Initial Soft Constraints", 
                                              "Initial Hard Constraints", "Final Soft Constraints", 
                                              "Final Hard Constraints", "Timestamp", 
                                              "feasible size", "Elapsed Time (seconds)", "Improvement"])
    
    # Write headers only if the file is empty
    if file.tell() == 0:
        writer.writeheader()

    for i in range(10):
        start_time = time.time()  # Record start time of the iteration

        # Step 1: Select a random file name
        file_name = instance_list[i % len(instance_list)] # 
        print(f"Selected file: {file_name}")
        instance1 = Instance(file_name)

        # Step 2: Initialize trial parameters
        trial_number = i
        print(f"Trial number: {trial_number}")

        # Step 3: Create populations
        r = np.random.randint(1, 100)
        feasible_solutions = create_population(instance1, size=r)
        infeasible_solutions = create_random_population(instance1, size=100 - r)
        total_population = feasible_solutions + infeasible_solutions
        print(f"Total population size: {len(total_population)}")

        # Step 4: Sort population based on criteria
        sorted_population = sorted(total_population, key=criteria)
        print("Population sorted.")

        # Step 5: Record initial constraints
        initial_soft_constraints = sorted_population[0].soft_penalty
        initial_hard_constraints = sorted_population[0].hard_penalty
        print(f"Initial soft constraints: {initial_soft_constraints}, Initial hard constraints: {initial_hard_constraints}")

        # Step 6: Apply genetic algorithm
        improved_population = genetic_algorithm(instance1, total_population, 100, 100, 50)
        print("Genetic algorithm applied.")

        # Step 7: Record final constraints
        final_soft_constraints = improved_population[0].soft_penalty
        final_hard_constraints = improved_population[0].hard_penalty
        print(f"Final soft constraints: {final_soft_constraints}, Final hard constraints: {final_hard_constraints}")

        # Calculate elapsed time for the iteration
        elapsed_time = time.time() - start_time

        # Step 8: Define the data structure for CSV
        data = {
            "File Name": file_name,
            "Trial Number": trial_number,
            "Initial Soft Constraints": initial_soft_constraints,
            "Initial Hard Constraints": initial_hard_constraints,
            "Final Soft Constraints": final_soft_constraints,
            "Final Hard Constraints": final_hard_constraints,
            "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "feasible size": r,
            "Elapsed Time (seconds)": elapsed_time,  # Add elapsed time to data
            "Improvement" : initial_soft_constraints-final_soft_constraints
        }

        # Write data to CSV
        writer.writerow(data)
        file.flush()
        print(f"Data saved to {csv_file}")



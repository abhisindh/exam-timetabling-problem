from utils import *
instance_list = ['car-s-91','car-f-92','ear-f-83', 'kfu-s-93', 'sta-f-83', 'tre-s-92', 'uta-s-92', 'yor-f-83']

instance_list += ['hec-s-92', 'lse-f-91', 'rye-s-93',   'ute-s-92'] # 'pur-s-93' is taking too much time
# Initialize CSV file
csv_file = "observations.csv"

# Write headers only if the file does not exist
header_written = False

try:
    with open(csv_file, mode='r'):
        header_written = True  # The file exists and we assume the header is already written
except FileNotFoundError:
    header_written = False  # The file doesn't exist, we need to write the header

for i in range(100):
    # Step 1: Select a random file name
    file_name = instance_list[i%len(instance_list)] #np.random.choice(instance_list)
    print(f"Selected file: {file_name}")
    instance1 = Instance(file_name)

    # Step 2: Initialize trial parameters
    trial_number = 1
    print(f"Trial number: {i}")

    # Step 3: Create populations
    r = np.random.randint(1,100)
    feasible_solutions = create_population(instance1, size=r)
    infeasible_solutions = create_random_population(instance1, size=100-r)
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

    # Step 8: Define the data structure for CSV
    data = {
        "File Name": file_name,
        "Trial Number": i,
        "Initial Soft Constraints": initial_soft_constraints,
        "Initial Hard Constraints": initial_hard_constraints,
        "Final Soft Constraints": final_soft_constraints,
        "Final Hard Constraints": final_hard_constraints,
        "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        "feasible : infeasible" : f"{r}:{100-r}"
        
    }

    # Step 9: Write data to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not header_written:
            writer.writeheader()  # Write header only if it's not already written
        writer.writerow(data)
        print("Data written to CSV.")

print(f"Data saved to {csv_file}")
from scripts import utils
import csv
import time

instance_list = utils.instance_list

if __name__=="__main__":
    
            
    
    for index, element in enumerate(utils.instance_list):
        print(f"Index {index}: {element}")
        
    idx = int(input("Enter the index of instance needed : "))
    instance = utils.Instance(instance_list[idx] , "data")
    r = int(input("Enter number of feasible elements needed in initial population : "))
    feasible_population = utils.create_population(instance, r)
    infeasible_population = utils.create_random_population(instance, 100-r)
    inital_population = feasible_population + infeasible_population
    
    criteria = lambda sol : sol.soft_penalty+ 1e9*sol.hard_penalty
    sorted_population = sorted(inital_population, key=criteria)
    initial_soft_constraints = sorted_population[0].soft_penalty
    initial_hard_constraints = sorted_population[0].hard_penalty
    print(f"Initial soft constraints: {initial_soft_constraints}, Initial hard constraints: {initial_hard_constraints}")

    input("Press Enter to start Genetic Algorithm \n>")
    start_time = time.time()
    improved_population = utils.genetic_algorithm(instance, inital_population)
    elapsed_time = time.time() - start_time
    final_soft_constraints = improved_population[0].soft_penalty
    final_hard_constraints = improved_population[0].hard_penalty
    print(f"Final soft constraints: {final_soft_constraints}, Final hard constraints: {final_hard_constraints}")
    
    data = {
        "File Name": instance.file_name,
        "Trial Number": 107,
        "Initial Soft Constraints": initial_soft_constraints,
        "Initial Hard Constraints": initial_hard_constraints,
        "Final Soft Constraints": final_soft_constraints,
        "Final Hard Constraints": final_hard_constraints,
        "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        "feasible size": r,
        "Elapsed Time (seconds)": elapsed_time,  # Add elapsed time to data
        "Improvement" : initial_soft_constraints-final_soft_constraints
        }
    csv_file = "docs/performance_tracking/main.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["File Name", "Trial Number", "Initial Soft Constraints", 
                                                "Initial Hard Constraints", "Final Soft Constraints", 
                                                "Final Hard Constraints", "Timestamp", 
                                                "feasible size", "Elapsed Time (seconds)", "Improvement"])
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(data)
        file.flush()
        print(f"Data saved to {csv_file}")
        

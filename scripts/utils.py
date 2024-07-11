import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import copy

instance_list = ['car-s-91',
                 'car-f-92',
                 'ear-f-83',
                 'hec-s-92',
                 'kfu-s-93',
                 'lse-f-91',
                 'pur-s-93',
                 'rye-s-93',
                 'sta-f-83',
                 'tre-s-92',
                 'uta-s-92',
                 'ute-s-92',
                 'yor-f-83']

data_folder = '../data'
class Instance():
    def __init__(self, file_name):
        self.file_name = file_name
        self.get_room_data()
        self.get_room_preference()
        self.get_course_data()
        self.get_dates()
        self.get_times()
        self.get_datetime_preference()
        self.get_student_data()
        self.max_violation_dict = {"overlap violation" : (self.s_c_matrix.sum(axis=1)-1).sum(),
                                   "capacity violation" : self.numRooms * self.numDateTime,
                                    "facility violation" : self.numCourses* self.numRooms,
                                   "proximity penalty" : np.triu(self.conflict_boolean, k=1).sum() * (2**5),
                                   "date time penalty" : 2**5 * self.numCourses,
                                   "room penalty" : 2**5 *self.numCourses
                                    }
        print(f"Created instance {file_name}.")
        print(f"number of courses = {self.numCourses}, number of students = {self.numStudents}")
        print(f"number of timeslots = {self.numDateTime}, number of rooms = {self.numRooms}")
        
    def get_room_data(self):
        self.room_df = pd.read_csv(f"{data_folder}/{self.file_name}/room_data.csv")
        self.capacityList = self.room_df['Capacity'].to_numpy(int)
        self.roomList = self.room_df['RoomId'].to_numpy()
        self.numRooms = len(self.roomList)
        self.r_f_matrix = self.room_df.drop(['RoomId', 'Capacity'], axis=1).to_numpy(int)
        self.roomIndex = {room: idx for idx, room in enumerate(self.roomList)}
        
    def get_room_preference(self):
        self.room_preference_df = pd.read_csv(f"{data_folder}/{self.file_name}/room_preference.csv")
        self.r_pref_matrix = self.room_preference_df.drop(['courseId'], axis=1).to_numpy(int)
        
    def get_course_data(self):
        self.course_df = pd.read_csv(f"{data_folder}/{self.file_name}/course_data.csv")
        self.courseList = self.course_df['courseId'].to_numpy()
        self.numCourses = len(self.courseList)
        self.courseIndex = {course: idx for idx, course in enumerate(self.courseList)}
        self.numStudentsList = self.course_df['numStudents'].to_numpy(int)
        self.c_f_matrix = self.course_df.drop(['courseId', 'numStudents'], axis=1).to_numpy(int)
        self.c_r_feasible = self.create_c_r_feasibility().astype(int)
        
    def get_dates(self):
        self.date_df = pd.read_csv(f"{data_folder}/{self.file_name}/dates.csv")
        self.dateList = self.date_df['Date'].to_numpy()
        self.tDayList = self.date_df['T-Day'].to_numpy()
        self.numDays = len(self.dateList)
        
    def get_times(self):
        self.time_df = pd.read_csv(f"{data_folder}/{self.file_name}/times.csv")
        self.timeList = self.time_df['Time'].to_numpy()
        self.numTime = len(self.timeList)
        self.dateTimeList = np.array([date + 'T' + time for date in self.dateList for time in self.timeList])
        self.numDateTime = len(self.dateTimeList)
        self.datetime_tday = {self.dateTimeList[i]:self.tDayList[i//self.numTime] for i in range(self.numDateTime)}
        
    def get_datetime_preference(self):
        self.datetime_preference_df = pd.read_csv(f"{data_folder}/{self.file_name}/datetime_preference.csv")
        self.d_pref_matrix = self.datetime_preference_df.drop(['courseId'], axis=1).to_numpy(int)              
        
    def get_student_data(self):
        self.student_df = pd.read_csv(f"{data_folder}/{self.file_name}/student_data.csv")
        self.studentList = self.student_df['rollNum'].to_numpy()
        self.student_df.set_index('rollNum', inplace=True)
        self.numStudents = len(self.studentList)
        self.studentIndex = {student: idx for idx, student in enumerate(self.studentList)}
        self.s_c_matrix =  self.create_s_c_matrix().astype(int)
        self.s_c_sparse = csr_matrix(self.s_c_matrix) # converting to sparse matrix due to large size
        self.conflict_matrix = self.s_c_sparse.T.dot(self.s_c_sparse).toarray()
        self.conflict_boolean = self.conflict_matrix.astype(bool)
        
    def create_s_c_matrix(self):
        s_c_matrix = np.zeros((self.numStudents, self.numCourses), dtype=int) # Student- Course boolean matrix from dataset
        for student in self.studentList:
            courses = self.student_df.at[student, 'coursesEnrolled'].split()
            s_id = self.studentIndex[student]
            for course in courses:
                c_id = self.courseIndex[course]
                s_c_matrix[s_id, c_id] = 1
        return s_c_matrix
    
    def create_c_r_feasibility(self):
        total_facilities_required = self.c_f_matrix.sum(axis=1)[:,np.newaxis]
        total_facilities_available = np.dot(self.c_f_matrix, self.r_f_matrix.T)
        self.c_r_facility = (total_facilities_available >= total_facilities_required).astype(int)
        
        capacity_required = self.numStudentsList[:,np.newaxis]
        capacity_available = self.capacityList
        self.c_r_capacity = (capacity_available >= capacity_required).astype(int)
        
        return self.c_r_facility & self.c_r_capacity

class Timetable():
    
    def __init__(self, instance, r_a, d_a):
        self.instance = instance
        self.r_a = r_a
        self.d_a = d_a
        self.get_instance_values()
        self.create_matrices()
        self.penalty_dict = {"overlap violation" : self.create_overlap_check().sum(),
                             "capacity violation" : self.create_capacity_check().sum(),
                             "facility violation" : self.create_facility_check().sum(),
                             "proximity penalty" : self.create_proximity_matrix().sum(),
                             "date time penalty" : self.create_date_penalty_matrix().sum(),
                             "room penalty" : self.create_room_penalty_matrix().sum()
                             }
        self.normalized_penalty = {name : self.penalty_dict[name]/ instance.max_violation_dict[name] for name in self.penalty_dict}
        self.soft_penalty = sum(list(self.normalized_penalty.values())[3:])
        self.hard_penalty = sum(list(self.normalized_penalty.values())[:3])
        
    def get_instance_values(self):
        self.numCourses = self.instance.numCourses
        self.numRooms = self.instance.numRooms
        self.numDateTime = self.instance.numDateTime
        
    def create_matrices(self):
        self.c_d_matrix = np.eye(self.numDateTime, dtype=int)[self.d_a]
        self.c_r_matrix = np.eye(self.numRooms, dtype=int)[self.r_a]
        #self.s_d_matrix =  (csr_matrix(self.instance.s_c_matrix).dot(csr_matrix(self.c_d_matrix))).toarray()  #np.dot(self.instance.s_c_matrix, self.c_d_matrix) #self.s_c_sparse.T.dot(self.s_c_sparse).toarray()
        
        self.overlap_check = self.create_overlap_check()
        self.capacity_check = self.create_capacity_check()
        self.facility_check = self.create_facility_check()
        self.proximity_matrix = self.create_proximity_matrix()
        
    def create_overlap_check(self):
        return ((np.dot(self.instance.conflict_boolean, self.c_d_matrix)-1) * self.c_d_matrix).sum(axis=1).astype(bool)
    
    def create_capacity_check(self):
        c_r_student = self.c_r_matrix * self.instance.numStudentsList[:,None] # in course-room binary matrix, change 1 to number of students writng that exma
        r_d_student = (csr_matrix(c_r_student).T.dot(csr_matrix(self.c_d_matrix))).toarray()                            #np.dot(c_r_student.T, self.c_d_matrix)
        return (r_d_student > self.instance.capacityList[:,np.newaxis] ).astype(int)
    
    def create_facility_check(self):
        required_facility = self.instance.c_f_matrix
        available_facility = self.instance.r_f_matrix[self.r_a, :]
        return (required_facility > available_facility).sum(axis=1).astype(bool)

    def create_proximity_matrix(timetable):
        # upper triangular matrix of size (numCourse x numCourse) that shows proximity of two courses if they have conflict
        tdayAssigned = timetable.instance.tDayList[timetable.d_a // timetable.instance.numTime]  # Calculate tday assignments
        diff_matrix = np.abs(tdayAssigned[:, None] - tdayAssigned)  # Calculate absolute differences
        proximity_matrix = np.triu(np.where(diff_matrix < 5, np.power(2, abs(5 - diff_matrix)), 0), k=1)  # Calculate proximity penalty
        return proximity_matrix*timetable.instance.conflict_matrix
    
    def create_date_penalty_matrix(timetable):
        c_d_preference = timetable.c_d_matrix* timetable.instance.d_pref_matrix
        return np.where((c_d_preference==5) | (c_d_preference==0), 0 , 2**(5-c_d_preference))
    
    def create_room_penalty_matrix(timetable):
        return timetable.c_r_matrix*np.where(timetable.instance.r_pref_matrix==5, 0, 2**(5-timetable.instance.r_pref_matrix))
    
    def display(self):
        data = {"courses" : self.instance.courseList,
                "room" : [self.instance.roomList[room] for room in self.r_a],
                "datetime" : [self.instance.dateTimeList[datetime] for datetime in self.d_a]
                }
        return pd.DataFrame(data)
def create_timetable(instance, heuristic='conflict_count'):
    numCourses = instance.numCourses
    numDateTime = instance.numDateTime
    numRooms = instance.numRooms
    shuffledDatetimeList = np.random.permutation(np.arange(numDateTime))
    
    # Initialize assignments
    d_a = np.full(numCourses, -1)
    r_a = np.full(numCourses, -1)
    r_d_capacity = np.tile(instance.capacityList, (numDateTime, 1))
    
    # Initialize course order based on heuristic
    
    if heuristic == 'conflict_count':
        conflict_count = instance.conflict_boolean.sum(axis=1)
        sorted_courses = np.argsort(conflict_count)[::-1]
    elif heuristic == 'conflict_degree':
        conflict_degree = instance.conflict_matrix.sum(axis=1)
        sorted_courses = np.argsort(conflict_degree)[::-1]
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    datetime_course = {dt: [] for dt in range(numDateTime)}
    
    # Assign courses to datetimes and rooms
    for course in sorted_courses:
        required_capacity = instance.numStudentsList[course]
        
        # Find available datetimes with no conflicts
        available_datetimes = []
        for dt in shuffledDatetimeList:
            if all(not instance.conflict_boolean[course, other_course] for other_course in datetime_course[dt]):
                available_datetimes.append(dt)
        
        if not available_datetimes:
            return None, None

        # Find available rooms for the selected datetime
        room_assigned = False
        for chosen_dt in available_datetimes:
            available_rooms = [room for room in range(numRooms) 
                               if instance.c_r_facility[course, room] and r_d_capacity[chosen_dt, room] >= required_capacity]
            if available_rooms:
                chosen_room = np.random.choice(available_rooms)
                d_a[course] = chosen_dt
                r_a[course] = chosen_room
                r_d_capacity[chosen_dt, chosen_room] -= required_capacity
                datetime_course[chosen_dt].append(course)
                room_assigned = True
                break
        
        if not room_assigned:
            return None, None
    
    return r_a, d_a

def create_population(instance, size=100, heuristic='conflict_count'):
    population = []
    for _ in tqdm(range(size)):
        r_a = d_a = None
        iteration = 0
        while r_a is None and iteration < 1000:
            r_a, d_a = create_timetable(instance, heuristic)
            iteration+=1
        timetable = Timetable(instance, r_a, d_a)
        population.append(timetable)
    return population

def create_random_population(instance, size=100):
    population = []
    for _ in tqdm(range(size)):
        d_a = np.random.randint(low=0, high=instance.numDateTime, size=instance.numCourses)
        r_a = np.random.randint(low=0, high=instance.numRooms, size=instance.numCourses)
        population.append(Timetable(instance, r_a, d_a))
    return population

def tournmanet_selection(population, pool_size, k=3):
    population = np.array(population)
    selected = []
    for i in range(pool_size):
        tournament_indices = np.random.choice(len(population), size=k, replace=False)
        tournament = population[tournament_indices]
        winner = min(tournament, key=lambda x: x.soft_penalty)
        selected.append(winner)
    return selected

def crossover(parent1, parent2, crossover_prob = 0.8):
    # if np.random.random() > crossover_prob:
    #     return parent1, parent2
    
    instance = parent1.instance
    numCourses = instance.numCourses
    ra_1 = parent1.r_a
    ra_2 = parent2.r_a
    da_1 = parent1.d_a
    da_2 = parent2.d_a

    r_crossover = np.random.randint(1, numCourses)
    ra_3 = np.concatenate((ra_1[:r_crossover], ra_2[r_crossover:]))
    ra_4 = np.concatenate((ra_2[:r_crossover], ra_1[r_crossover:]))

    d_crossover = np.random.randint(1, numCourses)
    da_3 = np.concatenate((da_1[:d_crossover], da_2[d_crossover:]))
    da_4 = np.concatenate((da_2[:d_crossover], da_1[d_crossover:]))

    return Timetable(instance, ra_3, da_3), Timetable(instance, ra_4, da_4)

def mutation(parent, mutation_rate = 0.5):
    # room
    instance = parent.instance
    mutated_room = parent.r_a.copy()
    random_numbers = np.random.rand(instance.numCourses)
    mutate_mask = random_numbers < mutation_rate
    mutated_room[mutate_mask] = np.random.randint(instance.numRooms, size = mutate_mask.sum())
    #datetime
    mutated_datetime = parent.d_a.copy()
    random_numbers = np.random.rand(instance.numCourses)
    mutate_mask = random_numbers < mutation_rate
    mutated_datetime[mutate_mask] = np.random.randint(instance.numDateTime, size = mutate_mask.sum())

    return Timetable(instance, mutated_room, mutated_datetime)

def weighted_penalty(timetable, weight= 100):
    return timetable.soft_penalty + timetable.hard_penalty*weight

def euclidean_distance(sol1, sol2):
    room_distance = np.sum((sol1.r_a - sol2.r_a)**2)
    datetime_distance = np.sum((sol1.d_a - sol2.d_a)**2)
    return np.sqrt(room_distance + datetime_distance)

def repair(input_solution, feasible_solutions, curr_depth=0, verbose=False, best_donor = None, max_depth = 10):
    instance = input_solution.instance
    
    print(verbose * f"Repairing: depth={curr_depth}, hard_penalty={input_solution.hard_penalty}\n", end='')
    
    if input_solution.hard_penalty == 0 or curr_depth > max_depth:
        print(verbose * f"Returning solution: depth={curr_depth}, hard_penalty={input_solution.hard_penalty}\n", end='')
        return input_solution
    if best_donor is None:
        best_donor = min(feasible_solutions, key=lambda sol: euclidean_distance(input_solution, sol))
    print(verbose * f"Best donor selected: {best_donor}\n", end='')
        
    for i in range(instance.numRooms):
        if input_solution.overlap_check[i] or input_solution.facility_check[i] or input_solution.capacity_check[input_solution.r_a[i], input_solution.d_a[i]]:
            r_a_copy = input_solution.r_a[:]
            d_a_copy = input_solution.d_a[:]
            d_a_copy[i] = best_donor.d_a[i]  # exchange date with best donor
            solution_copy = Timetable(instance, r_a_copy, d_a_copy)
            
            print(verbose * f"Trying date exchange: index={i}, donor date={best_donor.d_a[i]}\n", end='')
            
            if solution_copy.hard_penalty < input_solution.hard_penalty:  # check for reduction in hard penalty
                print(verbose * f"Date exchange improved solution: index={i}, new hard_penalty={solution_copy.hard_penalty}\n", end='')
                return repair(solution_copy, feasible_solutions, curr_depth+1, verbose, best_donor, max_depth=max_depth)
            else:
                r_a_copy[i] = best_donor.r_a[i]  # exchange room with best donor
                solution_copy = Timetable(instance, r_a_copy, d_a_copy)
                
                print(verbose * f"Trying room exchange: index={i}, donor room={best_donor.r_a[i]}\n", end='')
                
                if solution_copy.hard_penalty < input_solution.hard_penalty:  # check for reduction in hard penalty
                    print(verbose * f"Room exchange improved solution: index={i}, new hard_penalty={solution_copy.hard_penalty}\n", end='')
                    return repair(solution_copy, feasible_solutions, curr_depth+1, verbose, best_donor, max_depth=max_depth)
                else:
                    d_a_copy[i] = input_solution.d_a[i]  # revert date back to original
                    solution_copy = Timetable(instance, r_a_copy, d_a_copy)
                    
                    print(verbose * f"Reverting date: index={i}, original date={input_solution.d_a[i]}\n", end='')
                    
                    if solution_copy.hard_penalty < input_solution.hard_penalty:  # check for reduction in hard penalty
                        print(verbose * f"Reversion improved solution: index={i}, new hard_penalty={solution_copy.hard_penalty}\n", end='')
                        return repair(solution_copy, feasible_solutions, curr_depth+1, verbose, best_donor, max_depth=max_depth)
                
    print(verbose * f"No improvement found at depth {curr_depth}\n", end='')
    return input_solution

criteria = lambda sol : sol.soft_penalty 
def genetic_algorithm(instance, initial_population, generations, pop_size, pool_size):
    population = initial_population
    
    # Parameters
    crossover_rate = 0.8
    mutation_rate = 0.5
    tournament_size = 5
    elitism_count = 2

    for generation in tqdm(range(generations+1)):
        # Selection
        mating_pool = tournmanet_selection(population, pool_size, tournament_size)
        
        # Crossover and Mutation
        offsprings = []
        for i in range(0, pool_size-1, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[(i+1) % pool_size]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            
            if np.random.rand() < mutation_rate:
                child1 = mutation(child1, mutation_rate)
            if np.random.rand() < mutation_rate:
                child2 = mutation(child2, mutation_rate)
            
            if child1.hard_penalty:
                child1 = repair(child1, population)
            if child2.hard_penalty:
                child2 = repair(child2, population)
            
            offsprings.extend([child1, child2])
        
        # Combine and Select Next Generation
        combined_population = population + offsprings
        combined_population = sorted(combined_population, key=criteria)
        population = combined_population[:pop_size] #combined_population[:pop_size - elitism_count]
        
        # Add Elitism
        # elites = combined_population[:elitism_count]
        # population.extend(elites)
        
            
        if criteria(population[0]) == criteria(population[-1]):
            print(f"Generation {generation}: Best = {criteria(population[0])}, Worst = {criteria(population[-1])}")
            break
        
        if generation % (generations//10) == 0 or generation==generations :
            print(f"Generation {generation}: Best = {criteria(population[0])}, Worst = {criteria(population[-1])}")
            
    
    return population

if __name__ == '__main__':
    instance1 = Instance(instance_list[1])
    feasible_solutions = create_population(instance1,10)
    
    infeasible = crossover(feasible_solutions[0], feasible_solutions[1])[0]
    repair(infeasible, feasible_solutions,verbose=True, max_depth=50)

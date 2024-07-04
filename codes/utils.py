import numpy as np
import pandas as pd

class Timetable():
    def __init__(self, room_assignments=None, datetime_assignments=None):    
        self.room_assignments = room_assignments
        self.datetime_assignments = datetime_assignments
        self.c_r_matrix = self.create_course_room_matrix()
        self.c_d_matrix = self.create_course_datetime_matrix()
        self.s_d_matrix = np.dot(s_c_matrix, self.c_d_matrix)
        self.r_d_matrix = np.dot(self.c_r_matrix.T, self.c_d_matrix)
        self.facility_violations = 0
        self.capacity_violations = 0
        self.overlap_violations = 0
        self.room_violations = 0
        self.soft_penalty = self.calculate_soft_penalty()
        self.hard_penalty = self.facility_violations + self.capacity_violations+ self.overlap_violations + self.room_violations



    
    def is_feasible(self, verbose = False):
        output = True
        assigned_rooms = np.argmax(self.c_r_matrix, axis=1)
        required_facility = c_f_matrix
        available_facility = r_f_matrix[assigned_rooms, :]
        facility_check = required_facility <= available_facility
        
        if not facility_check.all():
            self.facility_violations = np.sum(~facility_check)
            print(verbose*f"Violations in facility check: {self.facility_violations} at {np.where(~facility_check)}\n", end="")
            # Optionally print the specific indices where the condition fails
            #print(np.where(~facility_check))
            output = False

        capacity_check = numStudent_list <= capacity_list[assigned_rooms]
        if not capacity_check.all():
            self.capacity_violations = np.sum(~capacity_check)
            print(verbose*f"Violations in capacity check: {self.capacity_violations} at {np.where(~capacity_check)}\n", end="")
            # Optionally print the specific indices where the condition fails
            # print(np.where(~capacity_check))
            output =  False

        if not np.all(self.s_d_matrix <= 1):
            self.overlap_violations = (np.sum(self.s_d_matrix > 1))/numCourses
            print(verbose*f"Violations in student overlap check: {self.overlap_violations} at {np.where(self.s_d_matrix > 1)}\n", end="")
            # Optionally print the specific indices where the condition fails
            # print(np.where(self.s_d_matrix > 1))
            output = False

        if not (self.r_d_matrix <= 1).all():
            self.room_violations = np.sum(self.r_d_matrix > 1)
            print(verbose*f"Violations in room overlap check: {self.room_violations} at {np.where(self.r_d_matrix > 1)}\n", end="")
            # Optionally print the specific indices where the condition fails
            # print(np.where(self.r_d_matrix > 1))
            output =  False
        return output
    
    def calculate_soft_penalty(self):
        self.is_feasible()
        hard_penalty = self.facility_violations + self.capacity_violations+ self.overlap_violations + self.room_violations
        weight = 1e6
        days_assigned = day[self.datetime_assignments]
        diff_matrix = days_assigned[:, None] - days_assigned
        proximity_matrix = np.where(diff_matrix < 5, np.power(2, np.abs(5 - diff_matrix)), 0)
        penalty = np.sum(np.triu(conflict_matrix * proximity_matrix, 1))
        return penalty #+ weight*hard_penalty

    def create_course_room_matrix(self):
        course_room_matrix = np.zeros((numCourses, numRooms))
        course_room_matrix[np.arange(numCourses), self.room_assignments] = 1
        return course_room_matrix
    
    def create_course_datetime_matrix(self):
        course_datetime_matrix = np.zeros((numCourses, numDatetime))
        course_datetime_matrix[np.arange(numCourses), self.datetime_assignments] = 1
        return course_datetime_matrix
      
    def display(self):
        data = {
            "courses" : course_list,
            "room" : [room_list[room] for room in self.room_assignments],
            "datetime" : [datetime_list[datetime] for datetime in self.datetime_assignments]
            
        }
        return pd.DataFrame(data)











def create_assignments():
    # Initialize room and datetime assignments with -1
    d_a = np.full(numCourses, -1)
    r_a = np.full(numCourses, -1)
    
    # Conflict mask and count of conflicts for each course
    conflict_mask = conflict_matrix.astype(bool)
    conflict_count = conflict_mask.sum(axis=1)
    
    # Sort courses by the number of conflicts in descending order
    sorted_courses = np.argsort(conflict_count)[::-1]
    course_index = 0
    
    while course_index < len(sorted_courses):
        course = sorted_courses[course_index]
        
        # Find feasible rooms for the course
        feasible_rooms = np.where(c_r_feasible[course])[0].tolist()
        
        # If no feasible rooms, move this course to the beginning of the list and restart
        if not feasible_rooms:
            print(f"No feasible rooms for course {course}, restarting...")
            sorted_courses = np.insert(np.delete(sorted_courses, course_index), 0, course)
            course_index = 0
            d_a.fill(-1)
            r_a.fill(-1)
            continue
        
        room_assigned = False
        while feasible_rooms:
            # Randomly select a room from feasible rooms and remove it from the list
            room = np.random.choice(feasible_rooms)
            feasible_rooms.remove(room)
            
            # Determine available datetimes with no conflicts and no exam in the selected room
            available_datetimes = np.array([
                dt for dt in range(numDatetime)
                if all((d_a[other_course] != dt or not conflict_mask[course][other_course])
                       for other_course in range(numCourses) if d_a[other_course] != -1) and
                   all((d_a[other_course] != dt or r_a[other_course] != room)
                       for other_course in range(numCourses) if r_a[other_course] == room)
            ])
            
            # If no available datetimes, try the next room
            if available_datetimes.size == 0:
                #print(f"No available datetime for course {course} in room {room}, trying next room...")
                continue
            
            # Randomly select an available datetime
            datetime = np.random.choice(available_datetimes)
            
            # Assign the room and datetime to the course
            r_a[course] = room
            d_a[course] = datetime
            room_assigned = True
            break
        
        if not room_assigned:
            # If no feasible room is found, put the course at the beginning of the list and restart
            print(f"No feasible datetime found for course {course}, restarting...")
            sorted_courses = np.insert(np.delete(sorted_courses, course_index), 0, course)
            course_index = 0
            d_a.fill(-1)
            r_a.fill(-1)
        else:
            course_index += 1
    
    return r_a, d_a






def calculate_weighted_penalty(timetable, weight=1e6):
    return timetable.soft_penalty + timetable.hard_penalty*weight
    
    
    
def tournament_selection(population, pool_size=50, tournament_size=3):
    # Convert population to a numpy array for easier manipulation
    population_array = np.array(population)
    
    selected = []
    
    for _ in range(pool_size):
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament = population_array[tournament_indices]
        winner = np.argmin([calculate_weighted_penalty(timetable) for timetable in tournament])
        
        # Append the selected timetable to the list
        selected.append(tournament[winner])
    
    
    return selected
    
    
    


def crossover(self, other):
    ra_1 = self.room_assignments
    ra_2 = other.room_assignments
    da_1 = self.datetime_assignments
    da_2 = other.datetime_assignments

    r_crossover = np.random.randint(1, numCourses)
    ra_3 = np.concatenate((ra_1[:r_crossover], ra_2[r_crossover:]))
    ra_4 = np.concatenate((ra_2[:r_crossover], ra_1[r_crossover:]))

    d_crossover = np.random.randint(1, numCourses)
    da_3 = np.concatenate((da_1[:d_crossover], da_2[d_crossover:]))
    da_4 = np.concatenate((da_2[:d_crossover], da_1[d_crossover:]))

    return Timetable(ra_3, da_3), Timetable(ra_4, da_4)

def mutate(self, mutation_rate=0.3):
    #room
    mutated_room = self.room_assignments.copy()
    random_numbers = np.random.rand(numCourses)
    mutate_mask = random_numbers < mutation_rate
    mutated_room[mutate_mask] = np.random.randint(numRooms, size = mutate_mask.sum())
    #datetime
    mutated_datetime = self.datetime_assignments.copy()
    random_numbers = np.random.rand(numCourses)
    mutate_mask = random_numbers < mutation_rate
    mutated_datetime[mutate_mask] = np.random.randint(numDatetime, size = mutate_mask.sum())

    return Timetable(mutated_room, mutated_datetime)


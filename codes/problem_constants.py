import pandas as pd
import numpy as np
import os

directory = 'data'
files = ['data/'+f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


# Dataframes from each file
date_df = pd.read_csv(files[0])
course_df = pd.read_csv(files[1])
room_df = pd.read_csv(files[4])
student_df = pd.read_csv(files[5])
time_df = pd.read_csv(files[6])


# List of each variable
date_list = date_df['Dates'].to_numpy()
course_list = np.array(course_df['Course Code'])
numStudent_list = course_df['numStudents'].to_numpy()
facility_list = course_df.columns[1:].to_numpy()
room_list = np.array(room_df['Room'])
capacity_list = room_df['Capacity'].to_numpy()
student_list = np.array(student_df['Roll number'])
time_list = np.array(time_df['Time-slot'])
datetime_list = np.array([s1+' '+s2 for s1 in date_list for s2 in time_list])

# Index dictionary of each variable
course_index = {course: idx for idx, course in enumerate(course_list)}
student_index = {student: idx for idx, student in enumerate(student_list)}
room_index = {room: idx for idx, room in enumerate(room_list)}
datetime_index = {datetime: idx for idx, datetime in enumerate(datetime_list)}
facility_index = {facility: idx for idx, facility in enumerate(facility_list)}

# Count of each variable
numCourses = len(course_list)
numStudents = len(student_list)
numRooms = len(room_list)
numDatetime = len(datetime_list)
numFacility = len(facility_list)




# Setting custom index
student_df.set_index('Roll number', inplace=True) #
room_df.set_index('Room', inplace=True)
course_df.set_index('Course Code', inplace=True)

# Problem matrices
c_f_matrix = course_df.drop(columns=['numStudents']).to_numpy() # Course-Facility  boolean matrix from dataset
r_f_matrix = room_df.drop(columns=['Capacity']).to_numpy() # Room- Facility boolean matrix from dataset
s_c_matrix = np.zeros((numStudents, numCourses)) # Student- Course boolean matrix from dataset
for student in student_list:
    courses = student_df.at[student, 'courses enrolled'].split()
    s_id = student_index[student]
    for course in courses:
        c_id = course_index[course]
        s_c_matrix[s_id, c_id] = True
        
#feasible rooms for course
total_facilities_required = np.sum(c_f_matrix, axis=1)
facility_comparison_matrix = np.dot(c_f_matrix, r_f_matrix.T) # Checks if room has facility c requires
c_r_facility = (facility_comparison_matrix >= total_facilities_required[:,np.newaxis]) 

num_students_per_course = course_df['numStudents'].to_numpy()
room_capacities = room_df['Capacity'].to_numpy()
c_r_capacity = (room_capacities >= num_students_per_course[:, np.newaxis])

c_r_feasible = c_r_capacity & c_r_facility # Course-Facility boolean matrix created that shows feasiblity of conducting an exam in a room

#conflict matrix
conflict_matrix = np.dot(s_c_matrix.T, s_c_matrix) # Conflic matix generating by multiplying s_c matrix with its transpose

# Date and which exam day is it (counting from start of exam)
from datetime import datetime
date_objects = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in datetime_list])
first_date = date_objects[0]
day = np.array([(date - first_date).days for date in date_objects])
#date_day = {datetime_list[i]: differences[i] for i in range(len(datetime_index))}













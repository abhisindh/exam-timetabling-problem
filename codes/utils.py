import pandas as pd
import numpy as np

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
        self.dateTimeList = np.array([date + 'T' + time for date in self.dateList for time in self.timeList])
        self.numDateTime = len(self.dateTimeList)
        
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
        
        return self.c_r_faciltiy & self.c_r_capacity

    
class Timetable():
    def __init__(self, instance, r_a, d_a):
        self.instance = instance
        self.r_a = r_a
        self.d_a = d_a
        self.get_instance_values()
        self.create_matrices()
        
    def get_instance_values(self):
        self.numCourses = self.instance.numCourses
        self.numRooms = self.instance.numRooms
        self.numDateTime = self.instance.numDateTime
        
    def create_matrices(self):
        self.c_d_matrix = np.eye(self.numDateTime, dtype=int)[d_a]
        self.c_r_matrix = np.eye(self.numRooms, dtype=int)[r_a]
        self.s_d_matrix = np.dot(self.instance.s_c_matrix, self.c_d_matrix)
        
        self.overlap_check = self.create_overlap_check()
        self.capacity_check = self.create_capacity_check()
        self.facility_check = self.create_facility_check()
        
    def create_overlap_check(self):
        return np.where(self.s_d_matrix > 1 , self.s_d_matrix-1, 0)
    
    def create_capacity_check(self):
        # in course-room binary matrix, change 1 to number of students writng that exam
        c_r_student = self.c_r_matrix * self.instance.numStudentsList[:,None] 
        r_d_student = np.dot(c_r_student.T, self.c_d_matrix)
        return (r_d_student > self.instance.capacityList[:,np.newaxis] ).astype(int)
    def create_facility_check(self):
        required_facility = self.instance.c_f_matrix
        available_facility = self.instance.r_f_matrix[self.r_a, :]
        return (required_facility > available_facility).astype(int)


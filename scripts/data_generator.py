import pandas as pd
import numpy as np
import os
import random
import string
import datetime

np.random.seed(42)
random.seed(42)

# Dictionary that contains the instance of data set and number of maximum exam days allowed
days_dict = {
 'car-s-91': 18,
 'car-f-92': 16,
 'ear-f-83': 12,
 'hec-s-92': 11,
 'kfu-s-93': 10,
 'lse-f-91': 11,
 'pur-s-93': 23,
 'rye-s-93': 14,
 'sta-f-83': 7,
 'tre-s-92': 12,
 'uta-s-92': 18,
 'ute-s-92': 7,
 'yor-f-83': 11
}

def file_name_to_path(file_name):
    return f"../data/RAW/{file_name}"

def file_name_to_folder(file_name):
    return f"../data/{file_name}"


def generate_random_string(total_length, num_digits=3, custom1='', custom2=''):
    """
    Generates a list of length `total_length` of random strings of format <custom1> <2-letter subject code><custom2> <numbers of `num_digits` length>
    """
    first_letter_combinations = ['MA', 'PY', 'CH', 'BI', 'EN', 'ST', 'HI', 'GO', 'CA']
    output = set()

    while len(output) < total_length:
        for letters in first_letter_combinations:
            if len(output) >= total_length:
                break
            number = ''.join(random.choices(string.digits, k=num_digits))
            new_string = custom1 + letters + custom2 + number
            output.add(new_string)
    
    return sorted(list(output)[:total_length])


def process_course_data(file_name):
    """
    Given an instace name of the dataset, it creates a csv file with columns : courseId	numStudents	computers	projector	whiteboard	internet	audio	printer	backup-power
    """
    global course_mapping
    global facility_list
    # Read the course data from the file
    file_path = f"{file_name_to_path(file_name)}.crs"
    df = pd.read_csv(file_path, sep=' ', header=None, names=['old_course_id', 'num_students'])

    # Rename course IDs based on subject prefixes
    
    course_mapping = generate_random_string(len(df))
    df['courseId'] = course_mapping
    
    df['numStudents'] = df['num_students']
    
    # Generate random facility columns
    facility_list = ['computers', 'projector', 'whiteboard', 'internet', 'audio', 'printer', 'backup-power']
    for facility in facility_list:
        df[facility] = [np.random.choice([0, 1], p=[0.8, 0.2]) for _ in range(len(df))]
        
    df.drop(['old_course_id', 'num_students'], inplace=True, axis=1)
    
    # Create a folder to store the output CSV
    output_folder = file_name_to_folder(file_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the processed data to CSV
    output_file = os.path.join(output_folder, 'course_data.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Processed course data saved to {output_file}")
    return df

def process_student_data(file_name):
    file_path = f"{file_name_to_path(file_name)}.stu"
    with open(file_path, 'r') as file:
        student_courses = file.readlines()
    course_list = [student.split() for student in student_courses]
    new_students = []
    for student in course_list:
        new_course = []
        for course in student:
            course_id = course_mapping[int(course)-1]
            new_course.append(course_id)
        new_students.append(" ".join(new_course))
    
    rollnum_list = generate_random_string(len(new_students),5,'23')
    
    df =pd.DataFrame(list(zip(rollnum_list, new_students)), columns=['rollNum', 'coursesEnrolled'])

    # Create a folder to store the output CSV
    output_folder = file_name_to_folder(file_name)
    #os.makedirs(output_folder, exist_ok=True)
    
    # Save the processed data to CSV
    output_file = os.path.join(output_folder, 'student_data.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Processed course data saved to {output_file}")
    return df
    
    
def create_room_data(file_name):
    """
    Creates room data based on the provided course data CSV file and saves it to a CSV file.

    Parameters:
    file_name (str): Name of the folder containing the course_data.csv file.

    Returns:
    pd.DataFrame: DataFrame containing room data with columns 'Room ID', 'Capacity',
                  'computers', 'projector', 'whiteboard', 'internet', 'audio', 'printer',
                  'backup-power'.
    """
    # Read course data CSV
    course_data = pd.read_csv(f'{file_name_to_folder(file_name)}/course_data.csv')
    data = course_data['numStudents']

    # Calculate histogram of course data
    a, b = np.histogram(data, bins=days_dict[file_name])

    # Adjust histogram data for room dataset
    num_rooms = days_dict[file_name]
    room_counts = np.ceil(a * num_rooms / sum(a)).astype(int)  # multiply by number of rooms and divide by course number to normalize 
    capacities = (np.ceil(b / 10) * 10)[1:].astype(int)  # Adjust bin edges for room capacities

    columns = ['RoomId', 'Capacity', 'computers', 'projector', 'whiteboard', 'internet', 'audio', 'printer', 'backup-power']

    # Create an empty DataFrame with specified columns
    room_df = pd.DataFrame(columns=columns)
    room_id = 1

    # Populate room_df with rooms based on capacities and room_counts
    for i in range(len(capacities)):
        if room_counts[i]!=0:
            for j in range(room_counts[i]):
                # Assign facilities randomly, ensuring at least one room has all facilities
                row = {
                    'RoomId': f'Room{room_id}',
                    'Capacity': capacities[i],
                    'computers': np.random.choice([0, 1], p=[0.2, 0.8]) if j != 0 else 1,
                    'projector': np.random.choice([0, 1], p=[0.2, 0.8]) if j != 0 else 1,
                    'whiteboard': np.random.choice([0, 1], p=[0.2, 0.8]) if j != 0 else 1,
                    'internet': np.random.choice([0, 1], p=[0.2, 0.8]) if j != 0 else 1,
                    'audio': np.random.choice([0, 1], p=[0.2, 0.8]) if j != 0 else 1,
                    'printer': np.random.choice([0, 1], p=[0.2, 0.8]) if j != 0 else 1,
                    'backup-power': np.random.choice([0, 1], p=[0.2, 0.8]) if j != 0 else 1
                }
                # Append row to room_df
                room_df = pd.concat([room_df, pd.DataFrame([row])], ignore_index=True)
                room_id += 1  # Increment room ID

    
    output_folder = file_name_to_folder(file_name)
    #os.makedirs(output_folder, exist_ok=True)
    
    # Save the processed data to CSV
    output_file = os.path.join(output_folder, 'room_data.csv')
    room_df.to_csv(output_file, index=False)
    
    print(f"Processed room data saved to {output_file}")

    return room_df

def create_dates(file_name):
    # Convert start_date to datetime object
    start_date = pd.to_datetime('2025-03-03')
    
    # Initialize lists for dates and t-day (distance from start_date)
    dates = []
    t_day = []
    
    # Start counting days from the start_date
    current_date = start_date
    
    # Number of days for the given instance
    num_days = days_dict[file_name]
    
    for _ in range(num_days):
        # Skip Saturdays and Sundays
        while current_date.weekday() >= 5:  # 5 and 6 are Saturday and Sunday
            current_date += datetime.timedelta(days=1)
        
        # Append date and t-day (distance from start_date)
        dates.append(current_date)
        t_day.append((current_date - start_date).days)
        
        # Move to the next day and increment t-day
        current_date += datetime.timedelta(days=1)
    
    # Create DataFrame from lists
    df = pd.DataFrame({'Date': dates, 'T-Day': t_day})
    # Create a folder to store the output CSV
    
    output_folder = file_name_to_folder(file_name)
    #os.makedirs(output_folder, exist_ok=True)
    
    # Save the processed data to CSV
    output_file = os.path.join(output_folder, 'dates.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Processed room data saved to {output_file}")
    
    return df


def create_times(file_name):
    # Define the times in 'hh:mm:ss' format
    times = ['10:00:00', '14:00:00']  # 10:00 AM and 2:00 PM in hh:mm:ss format
    
    # Create DataFrame
    df = pd.DataFrame({'Time': times})
    
    
    
    output_folder = file_name_to_folder(file_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the processed data to CSV
    output_file = os.path.join(output_folder, 'times.csv')
    df.to_csv(output_file, index = False)
    
    print(f"Processed times data saved to {output_file}")
    return df

def create_room_preference(file_name):
    # Load course and room data
    course_df = pd.read_csv(f"{file_name_to_folder(file_name)}/course_data.csv")
    room_df = pd.read_csv(f"{file_name_to_folder(file_name)}/room_data.csv")

    # Facility check
    c_f = course_df.drop(['numStudents', 'courseId'], axis=1).to_numpy(int)
    f_r = room_df.drop(['RoomId', 'Capacity'], axis=1).to_numpy(int).T
    c_r = np.dot(c_f, f_r)
    facility_check = c_r >= c_f.sum(axis=1)[:, np.newaxis]

    # Capacity check
    num_students = course_df['numStudents'].to_numpy(int)
    capacities = room_df['Capacity'].to_numpy(int)
    capacity_check = capacities >= num_students[:, np.newaxis]

    # Feasible rooms mask
    feasible_rooms = facility_check & capacity_check

    # Initialize preference matrix with zeros
    preference_matrix = np.zeros(feasible_rooms.shape, dtype=int)

    # Define probabilities for preferences [1, 2, 3, 4, 5]
    preferences = np.array([1, 2, 3, 4, 5])
    probabilities = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    # Randomly assign preferences where feasible
    random_preferences = np.random.choice(preferences, size=feasible_rooms.shape, p=probabilities)

    # Apply the mask to set preferences where feasible_rooms is True
    preference_matrix[feasible_rooms] = random_preferences[feasible_rooms]
    

    # Create DataFrame for better readability (optional)
    preference_df = pd.DataFrame(preference_matrix, 
                                 index=course_df['courseId'], 
                                 columns=room_df['RoomId'])

    # Save the processed data to CSV
    output_folder = file_name_to_folder(file_name)
    output_file = f"{output_folder}/room_preference.csv"
    preference_df.to_csv(output_file)

    print(f"Processed room preference data saved to {output_file}")
    return preference_df


def create_datetime_preference(file_name):
    # Load dates and times datasets
    dates_df = pd.read_csv(f"{file_name_to_folder(file_name)}/dates.csv")
    times_df = pd.read_csv(f"{file_name_to_folder(file_name)}/times.csv")

    # Create datetime combinations
    date_times = []
    for date in dates_df['Date']:
        for time in times_df['Time']:
            date_times.append(f"{date} {time}")

    # Create a preference matrix with zeros
    num_courses = len(course_mapping)
    num_datetimes = len(date_times)
    preference_matrix = np.zeros((num_courses, num_datetimes), dtype=int)

    # Define probabilities for preferences [1, 2, 3, 4, 5]
    preferences = np.array([1, 2, 3, 4, 5])
    probabilities = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    # Randomly assign preferences to each course-time combination
    random_preferences = np.random.choice(preferences, size=(num_courses, num_datetimes), p=probabilities)

    # Create DataFrame for better readability (optional)
    preference_df = pd.DataFrame(random_preferences, 
                                 index=course_mapping, 
                                 columns=date_times)
    preference_df.index.name = 'courseId'

    # Save the processed data to CSV
    output_folder = file_name_to_folder(file_name)
    output_file = f"{output_folder}/datetime_preference.csv"
    preference_df.to_csv(output_file, index=True)

    print(f"Processed datetime preference data saved to {output_file}")
    return preference_df



    
if __name__ == "__main__":
    #instance_list = ['car-s-91','car-f-92','ear-f-83', 'kfu-s-93', 'sta-f-83', 'tre-s-92', 'uta-s-92', 'yor-f-83']

    # instance_list = ['hec-s-92', 'lse-f-91', 'pur-s-93', 'rye-s-93',   'ute-s-92']
    # for instance in instance_list:
    #     process_course_data(instance)
    #     process_student_data(instance)
    #     create_room_data(instance)
    #     create_dates(instance)
    #     create_times(instance)
    #     create_room_preference(instance)
    #     create_datetime_preference(instance)
        
    pass

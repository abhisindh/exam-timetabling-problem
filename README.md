# Project Documentation




## Exam Timetabling Problem

This project aims to automate and optimize the scheduling of exams using advanced algorithms and data-driven approaches.

## Setup

1. Clone the repository:
```
git clone https://github.com/abhisindh/exam-timetabling-problem.git
cd exam-timetabling
```



1. Generate Data:
- Run scripts to generate room data, date datasets, and exam preferences.
```
cd codes
python data_generator.py
```

## Folder Structure

## Data Directory

The `data` directory is organized into subdirectories for each dataset instance used in the project:

- **RAW**: 
  - This folder contains the original datasets for each instance.
- **Generated Datasets**: 
  - Each of the following folders contains the processed and generated datasets for the respective instances:
    - `car-f-92`
    - `car-s-91`
    - `ear-f-83`
    - `hec-s-92`
    - `kfu-s-93`
    - `lse-f-91`
    - `pur-s-93`
    - `rye-s-93`
    - `sta-f-83`
    - `tre-s-92`
    - `uta-s-92`
    - `ute-s-92`
    - `yor-f-83`

These folders contain the generated datasets, split into separate files for ease of processing.

- **docs/**: Documentation folder.
- **codes/**: Scripts for data generation, algorithm implementation, and evaluation.

## Usage


1. Run algorithms to generate optimized exam timetables.

For detailed information, refer to `docs/algorithms.md` and other documentation files.

requirements.txt
makefile
Copy code
numpy==1.21.0
pandas==1.3.0
matplotlib==3.4.2
data_structure.md
markdown
Copy code
# Data Folder Structure

The `data` folder organizes datasets and generated data for the exam timetabling project.

## Instance 1 (`instance1/`)

### Original Datasets
- `course_data.csv`: Contains course information including number of students.
- `room_data.csv`: Generated dataset with room information, capacities, and facilities.

### Generated Data
- `exam_room_preference.csv`: Exam-room preferences based on facility requirements and capacities.
- `exam_time_preference.csv`: Exam-time preferences based on date-time combinations.

...

algorithms.md
markdown
Copy code
# Algorithms Used for Exam Timetabling

## 1. Feasibility Mask Generation

### Overview
- **Purpose**: Determine feasible exam-room combinations.
- **Method**: Calculate masks based on facility requirements and room capacities.

## 2. Preference Assignment

### Overview
- **Purpose**: Assign preferences to rooms and time slots for each exam.
- **Method**: Randomly assign preferences using predefined probability distributions.

## 3. Optimization Techniques

### Overview
- **Purpose**: Optimize exam schedules for minimal conflicts.
- **Method**: Implement genetic algorithms or heuristic approaches.

...

credits.md
markdown
Copy code
# Credits

## Libraries Used
- **NumPy**: Numerical computing library for array operations.
- **Pandas**: Data manipulation and analysis library.
- **Matplotlib**: Plotting library for visualizations.

## Data Sources
- Original datasets sourced from [source name or link].

## Contributors
- [Your Name]: Project lead and developer.
- [Contributor 1]: Algorithm design and implementation.
- [Contributor 2]: Data generation and analysis.

Summary
This documentation structure provides a clear overview of your project, its components, and how to use and understand it. Adjust the details based on your specific project requirements and contributions.
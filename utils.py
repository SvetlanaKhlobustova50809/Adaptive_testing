import json 

def get_code_dict(student_ids, pivot_code_id, code_df):
    code_dict = {}
    for student_id in student_ids:
        code_dict[student_id] = list(code_df[code_df['CodeStateID'] == student_id]['Code'].values)[0]
    # add pivot code
    code_dict[pivot_code_id] = list(code_df[code_df['CodeStateID'] == pivot_code_id]['Code'].values)[0]
    return code_dict

def get_problem_statement():
    with open('IRT_dataset/problem.txt') as f:
        problem_statement = f.read()
    return problem_statement

def get_pivot_code_id():
    with open('IRT_dataset/code67/pivot_code_id.txt') as f:
        pivot_code_id = f.read().strip()
    return pivot_code_id

def load_test_cases():
    with open('IRT_dataset/code67/test_cases.json') as f:
        test_cases = json.load(f)
    return test_cases

def down_IRT_dat():
    with open('IRT_dataset/code67/correct_outputs.json', 'r') as file:
        correct_answers = json.load(file)

    with open('IRT_dataset/code67/all_code_outputs.json', 'r') as file:
        student_answers = json.load(file)

    irt_results = {}

    for student, answers in student_answers.items():
        irt_results[student] = {}
        for key, answer in answers.items():
            if correct_answers[key] == answer:
                irt_results[student][key] = 1
            else:
                irt_results[student][key] = 0

    with open('IRT_dataset/code67/IRT_dataset.json', 'w') as file:
        json.dump(irt_results, file, indent=4)


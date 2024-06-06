'''
Loads the saved parameters from IRT_parameters folder
'''
import torch 

def load_irt_parameters():
    '''
    Load the saved parameters from IRT_parameters folder
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device == torch.device('cpu'):
        student_ability = torch.load('IRT/IRT_parameters/student_ability.pt', map_location=torch.device('cpu'))
        item_difficulty = torch.load('IRT/IRT_parameters/item_difficulty.pt', map_location=torch.device('cpu'))
    else:
        student_ability = torch.load('IRT/IRT_parameters/student_ability.pt')
        item_difficulty = torch.load('IRT/IRT_parameters/item_difficulty.pt')
    return student_ability, item_difficulty

def main():
    student_ability, item_difficulty = load_irt_parameters() 

    print(student_ability)
    print(item_difficulty)

if __name__ == '__main__':
    main()
from implement_irt import *

def main():
    CONSIDER_TEST_CASES = 36

    # set seed
    set_seed(37)

    # read dataset
    student_ids, outputs = read_dataset(CONSIDER_TEST_CASES) 

    # model parameters 
    num_students = len(outputs)
    # num_items = len(outputs[0])
    num_items = CONSIDER_TEST_CASES



    model, loss_fn, optimizer, num_epochs, device = get_model_info(num_students, num_items)

    # play with the model
    play_with_model(model)

    # get dataloader
    batch_size = 128
    train_dataloader = get_dataloader(batch_size, [i for i in range(num_students)], outputs)

    # train the model
    item_ids = [i for i in range(num_items)]
    model = train_IRT(item_ids, model, loss_fn, optimizer, num_epochs, device, train_dataloader)

    # Save the student ability and item difficulty separately
    save_dir = 'IRT/IRT_parameters'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.student_ability, '{:s}/student_ability.pt'.format(save_dir))
    torch.save(model.item_difficulty, '{:s}/item_difficulty.pt'.format(save_dir))



if __name__ == '__main__':
    main()
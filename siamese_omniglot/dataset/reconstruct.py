import os
import random
from glob import glob

import numpy as np
from tqdm import tqdm


def move_image_to_processed_dir(alpha_dirs, save_dir, desc):

    for alpha in tqdm(alpha_dirs, desc=desc):

        save_alpha_dir = save_dir + '/' + os.path.basename(alpha) + '_'

        for char in (os.listdir(alpha)):

            save_img_dir = (save_alpha_dir + char)
            os.makedirs(save_img_dir)

            char_imgs = os.path.join(alpha, char)

            for img_name in os.listdir(char_imgs):

                new_path = os.path.join(char_imgs, img_name)
                os.rename(new_path, os.path.join(save_img_dir, img_name))



def prepare_data(data_dir,seed):

    random.seed(seed)

    back_dir = data_dir + "/omniglot-py/images_background"
    eval_dir = data_dir + "/omniglot-py/images_evaluation"
    processed_dir = os.path.join(data_dir, "processed")

    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if any([True for _ in os.scandir(processed_dir)]):
        return train_dir, val_dir, test_dir

    # Move 10 of evaluation image for getting more train set.
    if len(glob(eval_dir + '/*')) >= 20:
        for d in random.sample(glob(eval_dir + '/*'), 10):
            os.rename(d, os.path.join(back_dir, os.path.basename(d)))

    back_alpha = sorted(glob(back_dir + "/*"))
    test_alpha = sorted(glob(eval_dir + "/*"))


    # Split background data into train, validation data and make test data
    train_alpha = np.random.choice(back_alpha, size=30, replace=False)
    train_alpha = [str(x) for x in train_alpha]
    val_alpha = [x for x in back_alpha if x not in train_alpha]


    move_image_to_processed_dir(train_alpha, train_dir, 'train')
    move_image_to_processed_dir(val_alpha, val_dir, 'val')
    move_image_to_processed_dir(test_alpha, test_dir, 'test')

    return train_dir, val_dir, test_dir
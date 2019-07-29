from nbr_rec.crnn import CRNN

def main():
    weight_save_path = "model/"
    train_img_dir = "train/"
    val_img_dir = "val/"
    img_size = (256, 32)
    num_classes = 11
    max_label_length = 26
    downsample_factor = 4
    batch_size = 64
    epochs = 100

    model = CRNN.build(img_size, num_classes, max_label_length)
    CRNN.train(model, train_img_dir, val_img_dir, weight_save_path,
               img_size, batch_size, max_label_length, downsample_factor, epochs)


if __name__ == '__main__':
    main()

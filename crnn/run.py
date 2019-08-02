from crnn.cnn_blstm_ctc import CNN_BLSTM_CTC


def main():
    model_save_path = "model/"
    img_size = (120, 46)
    num_classes = 11
    max_label_length = 26

    model = CNN_BLSTM_CTC.build(img_size, num_classes, max_label_length)
    CNN_BLSTM_CTC.train(model, "../dataset/imgs/", model_save_path, img_size,
                        batch_size=16, max_label_length=max_label_length, down_sample_factor=4, epochs=100)


if __name__ == '__main__':
    main()

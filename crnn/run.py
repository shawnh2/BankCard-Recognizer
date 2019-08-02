from crnn.cnn_blstm_ctc import CNN_BLSTM_CTC


def main():
    model_save_path = "model/"
    img_size = (256, 32)

    model = CNN_BLSTM_CTC.build()
    CNN_BLSTM_CTC.train(model, "../dataset/imgs/", model_save_path, img_size,
                        batch_size=32, max_label_length=26, down_sample_factor=4, epochs=100)


if __name__ == '__main__':
    main()

from crnn import CNN_BLSTM_CTC

def main():
    model_save_path = "model/"
    img_size = (256, 32)
    num_classes = 11
    max_label_length = 26
    down_sample_factor = 4
    epochs = 100

    model = CNN_BLSTM_CTC.build()
    CNN_BLSTM_CTC.train(model, "../dataset/imgs/", model_save_path, img_size, 32, 26, 4, 100)

if __name__ == '__main__':
    main()

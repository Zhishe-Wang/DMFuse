class args1():
    # training args

    epochs = 4  # "number of training epochs, default is 2"
    batch_size = 4  # "batch size for training, default is 4"

    data_ir_set ='E:/clip picture last/ir'
    data_vi_set = 'E:/clip picture last/vi'
    # data_ir_set = 'D:/ir'
    # data_vi_set = 'D:/vi'

    save_model_dir = "E:\qwq\ddpm-cd-master\models"  # "path to folder where trained model will be saved."
    save_loss_dir = "E:\qwq\ddpm-cd-master\loss"  # "path to folder where trained model will be saved."

    height = 256
    width = 256
    image_size = 256  # "size of training images, default is 224 X 224"
    cuda = 1  # "set it to 1 for running on GPU, 0 for CPU"


    lr = 1e-5  # "learning rate, default is 0.001"
    lr_light = 1e-5  # "learning rate, default is 0.001"
    log_interval = 1  # "number of images after which the training loss is logged, default is 500"
    log_iter = 1
    resume = None
    resume_auto_en = None
    resume_auto_de = None
    resume_auto_fn = None

    weight_SSIM = 10
    weight_Texture = 16
    weight_Intensity = 14



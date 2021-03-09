python main.py `
    --base_dir 'D:\Research\AV-SpeechEnhancement' `
    train_test `
    --data_dir 'E:\preprocessed\ss\dataset_03' `
    --model 'SS-AVCSE-L-fb_dataset_03_F', 'SS-AVCSE-L-fb_dataset_03_G' `
    --data_model 1, 1 `
    --data_train 'data_train'`
    --data_validation 'data_val' `
    --data_test 'data_test' `
    --batch-size 120 `
    --lr 5e-4 `
    --workers 12 `
    --test 'True' `
    --normalization 'False'
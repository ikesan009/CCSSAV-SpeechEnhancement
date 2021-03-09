python main.py `
    --base_dir 'D:\Research\AV-SpeechEnhancement\1.8' `
    train_test `
    --data_dir 'E:\preprocessed\ss_1.8\dataset_03' `
    --model 'SS-AVCSE-L-fb_dataset_03_F', 'SS-AVCSE-L-fb_dataset_03_G' `
    --data_model 0, 0 `
    --data_train 'data_train' `
    --data_validation 'data_val' `
    --data_test 'data_test' `
    --batch-size 120 `
    --lr 5e-4 `
    --workers 12 `
    --startepoch 1 `
    --endepoch 60 `
    --normalization 'False'
    
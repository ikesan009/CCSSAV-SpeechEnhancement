import os
import shutil

def make_dir(dir, format=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if format and os.path.exists(dir):
        shutil.rmtree(dir)


class AssetManager:

    def __init__(self, args):
        make_dir(args.base_dir)
        self.__preprocessed_dir = args.data_dir
        make_dir(self.__preprocessed_dir)
        self.__models_dir = os.path.join(args.base_dir, 'models')
        make_dir(self.__models_dir)
        self.__outputs_dir = os.path.join(args.base_dir, 'outputs')
        make_dir(self.__outputs_dir)

    def get_preprocessed_blob_path(self, data_name):
        dir_save = os.path.join(self.__preprocessed_dir, data_name)
        make_dir(dir_save)
        make_dir(os.path.join(dir_save, 'samples'))
        make_dir(os.path.join(dir_save, 'video_samples'))
        make_dir(os.path.join(dir_save, 'mixed_spectrograms'))
        make_dir(os.path.join(dir_save, 'speech_spectrograms'))
        make_dir(os.path.join(dir_save, 'ssed_spectrograms'))

        return dir_save

    def create_model(self, model_name):
        model_dir = os.path.join(self.__models_dir, model_name)
        make_dir(model_dir)

    def get_model_cache_path(self, model_name, epoch):
        model_dir = os.path.join(self.__models_dir, model_name)
        return os.path.join(model_dir, 'savedata_model_'+str(epoch+1)+'.pt')

    def get_normalization_cache_path(self, name_model, dir_data):
        name_normalization = os.path.split(dir_data)[1]  
        dir_model = os.path.join(self.__models_dir, name_model[0])
        return os.path.join(dir_model, 'normalization_'+name_normalization+'.pkl')

    def get_logger_cache_path(self, args):
        model_dir = os.path.join(self.__models_dir, args.model[0])
        if args.test == 'True':
            return os.path.join(model_dir, 'log_test.txt')
        else:
            return os.path.join(model_dir, 'log_train.txt')

    def get_copy_cache_path(self, args, path_base):
        dir_outputs = os.path.join(self.__outputs_dir, args.model[0])
        make_dir(dir_outputs)
        name_save = os.path.split(path_base)[1]
        return os.path.join(dir_outputs, name_save)

    def get_wav_cache_path(self, args, path_base, name):
        dir_outputs = os.path.join(self.__outputs_dir, args.model[0])
        name_save = os.path.split(path_base)[1]
        name_save = os.path.splitext(name_save)[0]
        name_save = name_save + '_' + name + '.wav'
        return os.path.join(dir_outputs, name_save)

'''
Author: Bappy Ahmed
Email: entbappy73@gmail.com
Date:05-Nov-2021
'''

from src.utils.all_utils import read_yaml, create_directory
import argparse
import os
import logging
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm



logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")



def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def embedding(config_path,params_path):
    
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config['artifacts']

    artifacts_dir = artifacts['artifacts_dir']
    pickle_format_data_dir = artifacts['pickle_format_data_dir']
    img_pickle_file_name = artifacts['img_pickle_file_name']

    feature_extraction_dir = artifacts['feature_extraction_dir']
    extracted_features_name = artifacts['extracted_features_name']

    raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
    feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)

    create_directory(dirs=[raw_local_dir_path,feature_extraction_path])
    
    pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)
    features_name = os.path.join(feature_extraction_path, extracted_features_name)

    data_path = params['base']['data_path']
    weight = params['base']['weights']
    include_tops = params['base']['include_top']


    model = ResNet50(weights= weight,include_top=include_tops,input_shape=(224,224,3))
    model.trainable = False

    model = tf.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    
    filenames = []

    for file in os.listdir(data_path):
        filenames.append(os.path.join(data_path,file))

    feature_list = []

    for file in tqdm(filenames):
        feature_list.append(extract_features(file,model))

    pickle.dump(feature_list,open(features_name,'wb'))
    pickle.dump(filenames,open(pickle_file,'wb'))



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info(">>>>> stage_01 started")
        embedding(config_path = parsed_args.config, params_path= parsed_args.params)
        logging.info("stage_01 completed!>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
    

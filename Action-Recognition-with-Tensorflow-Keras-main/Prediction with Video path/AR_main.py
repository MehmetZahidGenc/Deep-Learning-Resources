from ARpredictorClass import AR_predictor

MAX_SEQ_LENGTH = 30 # you should use whatever feature the model you are going to use is trained on
NUM_FEATURES = 2048 # you should use whatever feature the model you are going to use is trained on
IMG_SIZE = 320

classes = ['Basketball', 'BenchPres', 'Biking']

test_video_path = 'AcRecogDataset//test//Biking//v_Biking_g24_c05.avi'

predictor = AR_predictor(MAX_SEQ_LENGTH=MAX_SEQ_LENGTH, NUM_FEATURES=NUM_FEATURES, IMG_SIZE=IMG_SIZE, Seq_model_path='my_model.h5',
                         classes_of_model=classes, channel_size=3)


feature_extractor = predictor.build_feature_extractor()

predictor.sequence_prediction(path=test_video_path, feature_extractor=feature_extractor)
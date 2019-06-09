#Loading model architecture and whole model 

from keras.models import model_from_json 
from keras.models import load_model

#Load architecture 
json_file = open('model.json','r')
model_json = model_from_json(json_file.read())
model_json.summary()

#Load model 
#model1 = load_model("weightsk.h5")
#print("Loaded Model from disk")

# #compile and evaluate loaded model
# model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# loss,accuracy = model1.evaluate(X_test,y_test)
# print('loss:', loss)
# print('accuracy:', accuracy)
# # graph = tf.get_default_graph()


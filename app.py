import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
import pickle
from keras.preprocessing.image import load_img, img_to_array
app = Flask(__name__)


model = pickle.load(open('model_face_expression1.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    #query_index = np.random.choice(movie_features_df.shape[0])
    
    if(request.files['myCanvas']):
        #r=0
        query_index=request.files['myCanvas']
        #s=type(query_index)
        #image = cv2.imread(query_index,0)
        image= cv2.imdecode(np.fromstring(request.files['myCanvas'].read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(image, (48,48))
        #img = load_img(request.files['myCanvas'],target_size=(48, 48),grayscale=True)  
        x = img_to_array(img)  
        x = x.reshape((1,) + x.shape)
        h=model.predict([x])
        r=np.argmax(h)
        if(r==0):
            return render_template('index.html',prediction_text0="Гнев")
        elif(r==1):
            return render_template('index.html',prediction_text0="Отвращение")
        elif(r==2):
            return render_template('index.html',prediction_text0="Страх")
        elif(r==3):
            return render_template('index.html',prediction_text0="Радость ")
        elif(r==4):
            return render_template('index.html',prediction_text0="Безразличие")
        elif(r==5):
            return render_template('index.html',prediction_text0="Печаль")
        else:
            return render_template('index.html',prediction_text0="Удивление")
    
    
    
        
        
    else:
        return render_template('index.html',prediction_text0="Безразличие")
        
        
    
    
    
    
    #return render_template('index.html', prediction_text1=a,prediction_text2=b,prediction_text3=c,prediction_text4=d,prediction_text5=e,prediction_text6=f)



'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)

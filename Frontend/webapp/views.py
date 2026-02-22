from django.shortcuts import render
import pandas as pd
import numpy as np
from .models import eye
import os
import tensorflow as tf 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Create your views here.
homepage = 'index.html'
resultpage = 'result.html'
img_height, img_width = 224, 224

eyes = ['cardiovascular', 'No_cardiovascular']


def home(request):
    return render(request, homepage)


def result(request):
    if request.method == 'POST':
        m = int(request.POST['alg'])
        file = request.FILES['file']
        fn = eye(images=file)
        fn.save()
        path = os.path.join('webapp/static/image/', fn.filename())
        acc = pd.read_csv("webapp\Acc.csv")

        if m == 1:
            new_model = load_model(r"webapp\models\CNN.h5", compile=False)
            test_image = image.load_img(
                path, target_size=(img_height, img_width))
            test_image = image.img_to_array(test_image)
            test_image /= 255
            a = acc.iloc[m - 1, 1]

        elif m == 2:
            new_model = load_model(
                r"webapp\models\MobileNet.h5", compile=False)
            test_image = image.load_img(
                path, target_size=(img_height, img_width))
            test_image = image.img_to_array(test_image)
            test_image /= 255
            a = acc.iloc[m-1, 1]

        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        pred = eyes[np.argmax(result)]
        print(pred)
        
        if pred == "cardiovascular":
            mssg = "Remedies: Maintaining Stress Management, Healthy Diet, and Regular Physical Activity. In addition to these remedies, other lifestyle modifications such as quitting smoking, managing stress, getting enough sleep, and limiting alcohol intake can also contribute to better cardiovascular health. It's important to consult with a healthcare provider for personalized advice and recommendations based on individual health needs and medical history."
            
        else:
            mssg = "The individual is free from cardiovascular diseases"
            

        return render(request, resultpage, {'text': pred, 'path': 'static/image/'+fn.filename(),"mssg":mssg, 'a': round(a*100, 3)})
    return render(request, resultpage)

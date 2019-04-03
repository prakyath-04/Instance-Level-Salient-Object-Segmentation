"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
sys.path.insert(1, './src')
from crfrnn_model import get_crfrnn_model_def
import util
import numpy as np
import PIL
import ntpath 

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

inp_f = sys.argv[1]

def main():
    input_file = inp_f
    tmp = path_leaf(inp_f)
    print(tmp)
    output_file = '../saliency_maps/'+tmp

    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = 'crfrnn_keras_model.h5'

    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    img_data, img_h, img_w = util.get_preprocessed_image(input_file)
    probs = model.predict(img_data, verbose=False)[0, :, :, :]
    segmentation = util.get_label_image(probs, img_h, img_w)
    a = np.array(segmentation)
    print(np.shape(a))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if(a[i][j]>0):
                a[i][j] = 255
    segmentation = PIL.Image.fromarray(a)
    segmentation.save(output_file)


if __name__ == '__main__':
    main()

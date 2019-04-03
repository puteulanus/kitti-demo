
from car import *
import json
from tempfile import NamedTemporaryFile
import cv2
import hashlib
from urlparse import parse_qs
import os
import cgi

caffemodel='weights.caffemodel'
deploy_file='deploy.prototxt'

net = get_net(caffemodel, deploy_file, False)
transformer = get_transformer(deploy_file, None)
_, channels, height, width = transformer.inputs['data']
if channels == 3:
    mode = 'RGB'
elif channels == 1:
    mode = 'L'
else:
    raise ValueError('Invalid number for channels: %s' % channels)
    
result_images = {}

def app(environ, start_response):
    
    if environ['REQUEST_METHOD'] == 'OPTIONS':
        status = '200 OK'
        response_headers = [
            ('Content-type','text/plain'),
            ('Access-Control-Allow-Origin','*'),
            ('Content-Length', '0')
        ]
        start_response(status, response_headers)
        return iter([''])
        
    if environ['PATH_INFO'] == '/api/upload':
        d = cgi.FieldStorage(environ=environ, fp=environ['wsgi.input'], keep_blank_values=True)
        tmp = NamedTemporaryFile(mode='w+b', delete=True)
        tmp.write(d['file'].file.read())
        tmp.flush()
        image = [load_image(tmp.name, height, width, mode)]
        
        scores = forward_pass(image, net, transformer, batch_size=None)
        image_result = filter(lambda x: x[4] > 0, scores[0])
        
        img = cv2.imread(tmp.name, cv2.IMREAD_UNCHANGED)
        tmp.close()
        
        ht = img.shape[0] * 1.0 / height
        wt = img.shape[1] * 1.0 / width
        
        for i,x in enumerate(image_result):
            image_result[i][0] = int(x[0] * wt)
            image_result[i][1] = int(x[1] * ht)
            image_result[i][2] = int(x[2] * wt)
            image_result[i][3] = int(x[3] * ht)
            
        for x in image_result:
            cv2.rectangle(img,(x[0],x[1]),(x[2],x[3]),(0,255,0),3)
        
        img_file = NamedTemporaryFile(mode='w+b', delete=True)
        img_file.write(cv2.imencode('.jpg', img)[1].tostring())
        img_file.flush()
        
        img_file_c = NamedTemporaryFile(mode='w+b', delete=False)
        token = hashlib.md5(img_file_c.name).hexdigest()
        result_images[token] = img_file_c
        img_file_c.close()
        os.system('cjpeg -quality 70 -outfile '+ img_file_c.name + ' ' + img_file.name )
        img_file.close()
        
        status = '200 OK'
        ret = {
            'status' : 'ok',
            'data' : token
        }
        data = json.dumps(ret)
        response_headers = [
            ('Content-type','text/json'),
            ('Access-Control-Allow-Origin','*'),
            ('Content-Length', str(len(data)))
        ]
        start_response(status, response_headers)
        return iter([data])
    
    if environ['PATH_INFO'] == '/api/result':
        url_parameter = parse_qs(environ['QUERY_STRING'])
        token = url_parameter['token'][0]
        if not token in result_images:
            status = '404 NOT FOUND'
            response_headers = [
                ('Content-type','image/jpeg'),
                ('Access-Control-Allow-Origin','*'),
                ('Content-Length', '0')
            ]
            start_response(status, response_headers)
            return iter([''])
        else:
            status = '200 OK'
            with open(result_images[token].name, 'rb') as img_file:
                data = img_file.read()
            os.unlink(result_images[token].name)
            result_images.pop(token)
            response_headers = [
                ('Content-type','image/jpeg'),
                ('Access-Control-Allow-Origin','*'),
                ('Content-Length', str(len(data)))
            ]
            start_response(status, response_headers)
            return iter([data])
    
    status = '200 OK'
    response_headers = [
        ('Content-type','text/plain'),
        ('Access-Control-Allow-Origin','*'),
        ('Content-Length', '0')
    ]
    start_response(status, response_headers)
    return iter([''])
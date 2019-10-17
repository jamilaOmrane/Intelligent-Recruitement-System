#!/usr/bin/python3
from flask import Flask
from flask import jsonify
from flask import request
import scoring as sk 
import analyse.doc2ve_clustering as d2v



app = Flask(__name__)

profiles1 = [{'first_name': u'Hamza', 'last_name': u'Moslah', 'summary': u'Nothing to say', 'score': 100,
    'skills' : ['java', '0x', 'angular'], 'public_profile_url': u'url', 'num_connections': 1000,
    'educations': [{'degree': u'Engireeing', 'start_date': u'2015', 'end_date': u'2018', 'field_of_study': u'software',
    'school_name': u'INSAT'}], 'positions': [{'company_name': u'Talan Tunisie', 'is_currrent': True,
    'start_date': u'2018', 'end_date': u'', 'summary': u'.....', 'title': u'0x Protocol'}],
    'industry': u'Blockchain', 'location': u'Tunis'}]

@app.before_request
def option_autoreply():
    """ Always reply 200 on OPTIONS request """
    print(request.headers)
    print(request.method)

    if request.method == 'OPTIONS':
        resp = app.make_default_options_response()

        headers = None
        if 'ACCESS_CONTROL_REQUEST_HEADERS' in request.headers:
            headers = request.headers['ACCESS_CONTROL_REQUEST_HEADERS']

        h = resp.headers

        # Allow the origin which made the XHR
        h['Access-Control-Allow-Origin'] = "*"
        # Allow the actual method
        h['Access-Control-Allow-Methods'] = request.headers['Access-Control-Request-Method']
        # Allow for 10 seconds
        h['Access-Control-Max-Age'] = "10"

        # We also keep current headers
        if headers is not None:
            h['Access-Control-Allow-Headers'] = headers

        return resp


@app.after_request
def set_allow_origin(resp):
    """ Set origin for GET, POST, PUT, DELETE requests """

    h = resp.headers

    # Allow crossdomain for other HTTP Verbs
    if request.method != 'OPTIONS' and 'Origin' in request.headers:
        h['Access-Control-Allow-Origin'] ="*"


    return resp

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/api/v1.0/profiles/',  methods=['GET'])
def list_profiles():
    """profiles = d2v.main_process("aaaaa")
    return jsonify(profiles)"""
    desired_profile = {"title":request.args.get('title') ,"summary":request.args.get('summary'), "skills":request.args.getlist('skill')}
    profiles = sk.scoring_profile(desired_profile)
    return jsonify(profiles)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')

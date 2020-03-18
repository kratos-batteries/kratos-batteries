from __future__ import print_function
import pandas as pd
import json
import sys
import requests

_api_version = str('0.0.1')

class MagpieServer:
    """Object to store how to connect to a server running Magpie"""
    
    _url = None
    """ URL of server """
    _models = None
    """ Cached information about models held by this server. """
    
    def __init__(self, url="http://josquin.northwestern.edu:4581/"):
        """Create a connection to a Magpie server. Defaults to
        connecting with a server running at Northwestern University
        hosted by the Wolverton group.
        
        :param url: URL of server
        :param port: Port number"""
        
        self._url = url
        
        # Test whether API versions agree
        self.api_version()
        
    def _make_request(self, path, data=None, method='get'):
        """Perform a request. Handles making error messages
        
        :param path: str, path of request
        :param data: Any data to be passed as JSON
        :return: requests.Request"""
        
        r = requests.request(method=method, url=self._url + path, 
            data=data)
        
        # Check error status
        if r.ok:
            return r
        else:
            raise Exception('Request failed. Status = %d. Reason = %s'%(r.status_code, r.reason))
    
    def api_version(self):
        """Get the API version of the server. 
        
        Prints error message of that version is different than what is supported
        by this wrapper.
        
        :return: API version"""
        
        # Make the requested
        r = self._make_request("server/version")
        v = r.content
        
        # If Python 3, convert to string
        if isinstance(v, bytes):
            v = v.decode()
        
        # Check whether it agrees with version of this wrapper
        if _api_version != v:
            print("WARNING: API version of Magpie server different than wrapper: %s!=%s"%(_api_version, v), file=sys.stderr)
        
        return v
        
    def status(self):
        """Get the status of the Magpie server
        
        :return: Status of server as dict"""
        
        return self._make_request("server/status").json()
        
    def models(self):
        """Get information about models held by this server
        
        :return: dict, Information about all the models"""
        
        if self._models is None:
            self._models = self._make_request("models").json()
        
        return self._models
        
    def get_model_info(self, name):
        """Get information about a specific model
        
        :param name: str, name of model
        :return: dict, information about a model"""
        
        if self._models is None or name not in self._models:
            r = self._make_request("model/%s/info"%name)
            return r.json()
        else:
            return self._models[name]
        
        
    def generate_attributes(self, name, entries):
        """Generate attributes that serve as input to a certain model
        
        :param name: str, name of model
        :param entries: list, list of entries to be run (as strings)
        :return: Pandas array, where [i,j] is attribute j of entries[i]"""
        
        # Package the request
        data = dict(entries=json.dumps(dict(entries=[dict(name=e) for e in entries])))
        r = self._make_request("model/%s/attributes"%name, data=data, method='POST')
        
        # Compile entries into numpy array
        results = r.json()
        attrs = pd.DataFrame([x['attributes'] for x in results['entries']], 
            columns=results['attributes'])
        return attrs
    
    def run_model(self, name, entries):
        """Run a particular model.
        
        :param name: str, Name of model to be run
        :param entries: list, list of entries to be run (as strings)
        :return: Predicted values. Also generates the probabilities 
        for membership in each class for classifier models
            Second column is always the predicted value as a number."""
        
        # Get the information about this model
        model_info = self.get_model_info(name)
        
        # Check whether it is a regression model
        reg = model_info['modelType'] == "regression"
        
        # Run the model 
        data = dict(entries=json.dumps(dict(entries=[dict(name=e) for e in entries])))
        r = self._make_request("model/%s/run"%name, data=data, method='POST')
        
        # Generate the output dataframe
        results = r.json()
        if reg:
            return pd.DataFrame(list(zip(entries,[x['predictedValue'] for x in results['entries']])),
                columns=['Entry']+['%s (%s)'%(model_info['property'], model_info['units'])])
        else:
            # Get probabilities
            classes = model_info['units']
            probs = []
            for c in classes:
                probs.append([e['classProbabilities'][c] for e in results['entries']])
            
            return pd.DataFrame(list(zip(entries,[x['predictedValue'] for x in results['entries']],
                    [x['predictedClass'] for x in results['entries']], *probs)),
                    columns=['Entry']+['Class','ClassName']+['P(%s)'%c for c in classes])
        

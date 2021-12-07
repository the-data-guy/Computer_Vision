import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import json
from oauth2client.client import GoogleCredentials
import requests

# TODO: Parametrize these, rather than hard-coding
pub_sub_topic = 'projects/kubeflow-1-0-2/topics/inference_images'
subscription_id = 'projects/kubeflow-1-0-2/subscriptions/my_subscription'
PROJECT = "kubeflow-1-0-2"
REGION = "us-central1"
ENDPOINT_ID = "6569494015031377920"

class ModelPredict:
    def __init__(self, project, region, endpoint_id):
        self._api = "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/endpoints/{}:predict".format(
                                                                                                                   region,
                                                                                                                   project,
                                                                                                                   region,
                                                                                                                   endpoint_id
                                                                                                                  )   
        
    def __call__(self, filenames):        
        token = GoogleCredentials.get_application_default().get_access_token().access_token
        if isinstance(filenames, str):
            # Only one element, put it into a batch of 1.
            data = {
                    "instances": [
                                  {"filenames": filenames}
                                 ]
                   }
        else:
            data = {
                    "instances": []
                   }
            for f in filenames:
                data["instances"].append({
                                          "filenames" : f
                                        })
        # print(data)
        headers = {"Authorization": "Bearer " + token }
        response = requests.post(self._api,
                                 json=data,
                                 headers=headers)
        response = json.loads(response.content.decode("utf-8"))
        
        if isinstance(filenames, str):
            result = response["predictions"][0]
            result["filename"] = filenames
            yield result
        else:
            for (a,b) in zip(filenames, response["predictions"]):
                result = b
                result["filename"] = a
                yield result

pipeline_options = PipelineOptions(
                                   streaming=True,  # required for Beam connector with pub/sub
                                  )
                
with beam.Pipeline(options=pipeline_options) as p:
    (p 
     | "getinput" >> beam.io.ReadFromPubSub(subscription=subscription_id)
     | "batch" >> beam.BatchElements(min_batch_size=2,
                                     max_batch_size=3)
     | "getpred" >> beam.FlatMap(ModelPredict(PROJECT,
                                              REGION,
                                              ENDPOINT_ID))
     | "write" >> beam.Map(print)
    )

import os, requests
requests.delete(os.environ['NSML_METRIC_API']).raise_for_status()


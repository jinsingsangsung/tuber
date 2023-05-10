import os, requests
import traceback

def set_nsml_reschedule():
    try:
        api_host = os.environ["NSML_RUN_METADATA_API"]
        api_secret = os.environ["NSML_RUN_SECRET"]
        requests.put(f"{api_host}/v1/rescheduled", headers={"X-NSML-Run-Secret": api_secret}, json={"rescheduled": True}).raise_for_status()
    except:
        # Sometimes, the HTTP request might fail, but the training process should not be stopped.
        traceback.print_exc()


def unset_nsml_reschedule():
    try:
        api_host = os.environ["NSML_RUN_METADATA_API"]
        api_secret = os.environ["NSML_RUN_SECRET"]
        requests.put(f'{api_host}/v1/rescheduled', headers={'X-NSML-Run-Secret': api_secret}, json={'rescheduled': False}).raise_for_status()		
    except:
        # 네트워크 에러가 일어나는 경우 등 대비
        traceback.print_exc()
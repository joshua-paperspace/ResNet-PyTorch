import sys, getopt

def main(argv):

    try:
        opts, args = getopt.getopt(argv,"i:",["id="])
    except:
        print("Could not get any passed arguments.")
    for opt, arg in opts:
        if opt in ('-i', '--id'):
            model_id = arg

    print(model_id)


    deployment_scirpt = '''\
image: nvcr.io/nvidia/pytorch:21.10-py3
port: 8501
command:
  - /bin/sh
  - '-c'
  - |
    cd /opt/repos/repo
    pip install -r requirements.txt
    streamlit run app.py
models:
  - id: {id}
    path: /opt/models
repositories:
  dataset: dsrp37o09pa0h8u
  mountPath: /opt/repos
  repositories:
    - url: https://github.com/joshua-paperspace/ResNet-PyTorch
      name: repo
      ref: main
resources:
  replicas: 1
  instanceType: C4'''.format(id=model_id)

    with open("./deployment.yaml", "w") as o:
        o.write(deployment_scirpt)

    return None

if __name__ == '__main__':
	main(sys.argv[1:])
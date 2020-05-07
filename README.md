# DEBUG : Definition Extraction for Building Useful Glossaries

This is a project in which definition extraction is explored. Definition extraction is broken into 2 processes :
1. Whether sentences contain a definition
2. Tagging the tokens of a sentence containing the definition.

## Datasets
1. WCL corpus [http://lcl.uniroma1.it/wcl/]
2. DEFT corpus [https://github.com/adobe-research/deft_corpus]

The source code related to the training of the models can be found in *notebooks* and the pretrained models on datasets
can be found in *models*.

The end to end pipeline can be found in the *definition_extractor.py* script.

## Setting up the server - Ubuntu-16.04 with GPU Tesla K80
1. Install python3.6 
  ```
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install python3.6
  ```
2. Install cuda 10.1 - [Reference](https://developer.nvidia.com/cuda-10.1-download-archive-base)
  ```
  wget --header="Connection: keep-alive" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.105-1_amd64.deb" -c -O 'cuda-repo-ubuntu1604_10.1.105-1_amd64.deb'
  sudo dpkg -i cuda-repo-ubuntu1604_10.1.105-1_amd64.deb
  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  sudo apt-get update
  sudo apt-get install cuda
  ```
3. Install conda - Reference [1](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04) and [2](https://askubuntu.com/a/507666/1005427)
  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
  bash Anaconda3-2020.02-Linux-x86_64.sh
  source ~/.bashrc
  ```
4. Install git lfs - [Reference](https://stackoverflow.com/a/48734334/8293309)
  ```
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get install git-lfs
  ```
5. Clone github repository (The traditional git clone didn't appear to work) - [SO Answer](https://stackoverflow.com/questions/48392440/git-lfs-clone-vs-git-clone)
  ```
  git lfs clone https://github.com/aadarshsingh191198/glossary_builder.git
  ```
6. Create conda environment 
  ```
  conda create --name tf1_15 python=3.6
  conda activate tf1_15
  ```
7. Install tensorflow 1.15 - [Why use conda for tensorflow?](https://towardsdatascience.com/stop-installing-tensorflow-using-pip-for-performance-sake-5854f9d9eb0c)
  ```
  conda install tensorflow==1.15
  ```
8. Install remaining dependencies
  ```
  pip install -r requirements.txt
  ```
9. Install spacy's small English model
  ```
  python -m spacy download "en_core_web_sm"
  ```
10. Run server 
  ```
  python app.py
  ```
11. Setting up production server - [Reference](https://towardsdatascience.com/deploying-a-custom-ml-prediction-service-on-google-cloud-ae3be7e6d38f)
  ```
  sudo apt-get install nginx-full
  sudo /etc/init.d/nginx start
  
  # remove default configuration file
  sudo rm /etc/nginx/sites-enabled/default
  
  # create a new site configuration file
  sudo touch /etc/nginx/sites-available/flask_project
  sudo ln -s /etc/nginx/sites-available/flask_project /etc/nginx/sites-enabled/flask_project
  
  # Edit configuration file
  sudo nano /etc/nginx/sites-enabled/flask_project
  
  #Copy and paste the following code 
    server {
      location / {
          proxy_pass http://0.0.0.0:8000;
      }
  }
  ```
  Other reference links:
  
  1. Using conda with pip - [1](https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/pip-interoperability.html) and [2](http://datumorphism.com/til/programming/python/python-anaconda-install-requirements/)
  2. [Exposing GCE ports](https://serverfault.com/questions/831273/unable-to-reach-a-python-flask-enabled-web-server-on-gce)
  3. [Increasing GPU Quota on GCE](https://stackoverflow.com/a/53678838/8293309). Sidenote - Checkout the comment section too.

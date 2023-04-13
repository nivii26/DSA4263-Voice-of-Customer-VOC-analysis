# DSA4263: Voice of the Customer (VoC) Analysis

# Introduction

The Voice of the Customer (VoC) is a methodology used to capture customers’ needs, requirements, and perceptions about products or services. In this project, we seek to develop and end-end solution for Voice of Customer analysis to provide insights to end business users using a public customer feedback dataset.

## Installation

1. **Docker**

Run  

```docker build -t api .```  
```docker run -d -p 5000:5000 api```

2. **Virtual Environment**

- Create a virtual environment of your choice; eg:
```virtualenv venv```
- Activate virtual environment & Install dependencies
```source venv/Scripts/activate```
```pip install -r requirements.txt```
- Run the API with
```uvicorn --app-dir=./root/src main:app --port 5000```

The app is hosted on ```localhost (127.0.0.1)``` at port ```5000```. API documentation is available at ```127.0.0.1:5000/docs```

![image](./root/src/assets/swagger.jpg)

Some usage examples can be found in the [api_demo notebook](https://https://github.com/nivii26/DSA4263-Voice-of-Customer-VOC-analysis/tree/main/root/ipynb/api_demo.ipynb)

## TODO: Fill in markdown

## Task Breakdown

TASK 1 : Sentiment Analysis

TASK 2 : Topic Modeling

To train the topic model from outside of the root directory, run: ```python -m root.src.model.tm.tm_train```. This will train and deposit a topic model into root/models/tm/. Configuration of the training hyperparameters is specified in root/src/model/tm/config.yml.

TASK 3 : Visualisation

To load the visualisation locally, ensure that the packages streamlit and Circlify (in the requirements.txt) are already installed. Run ```streamlit run root/src/visualization/visualize.py``` in the terminal. If successful, the visualisation will be opened in the defualt browser.

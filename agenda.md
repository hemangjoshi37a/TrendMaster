# New agenda

## this is a github python repo in which user can install this using pip. to use this repo you have to load the data using `data_loader.py` and train the model using `trainer.py` file and infer the trained model using `inferencer.py` file. all the files should contain appropriate comments and docstring for better readability of the code

## the `data_loader.py` file should have three functionalites in it's code , first is to autheniutacte user by prompitng for entering user auth details of zerodha account, then ask for the stock symbol for which a user is using this repo. then download the data using zerodha kite class. then saving it approriately in a local file for furhter model training usecases. here the file `notebook62326ade97_1X.ipynb` and `inference_notebook-Copy1.ipynb` can be referenced for this

## the `trainer.py` file should have code for this . this file should contain code for loading local file that is saved by the `data_loader.py` file's code execution , if not found then give assering error for this . then asking user for transofrmer parameters for training the model. then one method for starting the training process with live plot display and saving the plots . after completion of training of this model it should saved locally with well defined name

## here in the file `inferencer.py` it should have code for loading the model that is saved by the file `trainer.py` and then for inferring this model it should ask for authenticating using zerodha using preovious files if not already logged in. and then ask in a prompt for inferring the stock symbol for this . then  after getting name of the symbol that the user want to infer it should get data and load the model and run inference code and how the results in a plot and analytical table

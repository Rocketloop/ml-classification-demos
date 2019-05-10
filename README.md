# Rocketloop - Machine Learning Classification Demos

The dataset in `data_files` is from here: https://archive.ics.uci.edu/ml/datasets/Adult.

To run the script we recommend creating a virtual environment to install the required packages.
Do so by typing `virtualenv venv` in your terminal in the directory where you want to install the environment.
The next step will be to start the virtual environment with `source venv/bin/activate`.
Once you are in the virtual environment enter `pip install -r requirements.txt` to install the required packages.
After the install you are good to run `python3 classifiers.py`.
For any changes of the data set used for training or testing of the models you have to manually change the code.
To do so use an editor of your choice and navigate to the main function and edit the values for 'files'. The first file is the train data and the second file is the test data.
When you want to leave the virtual environment simply type `deactivate`.

If you speak german we highly recommending reading our blog. You will find helpful information and insights about what the script does and how it works. Simply hot the following link: https://rocketloop.de/blog/

We included reduced data sets in the data_files folder to compare training times. The files were created using the datasplitter.py, a small script which splits a dataset into a 1/4 train set and 3/4 test set. It may be a bit buggy so verify the outcome manually if you want to use it. Usage: Change the value for 'file' to the file you want to split. Change the value for 'filee' to the file you want to store the train data in. Change the value for 'fileee' to the file you want to store the test data in.   



# House Prices Kaggle competition
My efforts to work on the house prices data prediction competition and create the most accurate prediction I can

## Description
The house prices competition is a data prediction contest on Kaggle in which contenders try to predict the most accurate house price possible for a given house based on data points such as number of rooms, balcony space, date it was built, etc. In order to win, the given dataset needs to be used meticulously to generate a data prediction algorithm that can take into account all of those details to the highest degree of accuracy possible.
Further competition details can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

This machine learning problem is my first real-world application of the skills I have learnt in Andrew Ng's online Stanford University machine learning course, and as such I want to use Octave programming to further deepen my understanding of the core mathematics of machine learning. As such I can continue learning from the course by seeing how to apply my knowledge to real, large datasets. If I manage to get a reasonable score on this competition, it will serve as proof that I have properly understood the course and am on my way to mastering the core concepts of Machine Learning

## Programs used
**Jupyter (and hence Python)** - to process the given csv files and convert it to Octave-friendly data, by using the very useful functions such as the One-Hot encoder. Then outputs a txt file that Octave can import
**Octave** - for the actual Machine Learning, to go through the data, determine the optimum regularization parameters, and find the best theta to fit the function. Will output the prediction data. 

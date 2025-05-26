# Predicting Titanic Passengers' Chances of Survival Using Random Forest

Welp, I had a little too much time on my hands this time, and decided that it would be a good use of my free time to participate in a Kaggle competition. 

The competition itself is an ongoing one and has been open for quite some time, but I doubt that it's going to close anytime soon. (FYI, this is a prize-less competition)

Despite the lack of incentive, I figured that this would be a good competency test to see if I could apply everything I've learned about ML so far. 

## Information

> The sinking of the Titanic is one of the most infamous shipwrecks in history.
<br>On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
<br>While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
<br>In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
<br>https://www.kaggle.com/competitions/titanic/overview

## Walkthrough

Okay, the subheading is a little misleading, because it's not much of a walkthrough as it is about an explanation (or several) of what I did and why I did them. 
One of the first things I realized much later when I was working through the challenge was the fact that the organizers hadn't actually separated training data from the test data.
`test.csv` as it turns out isn't a dataset used to evaluate the model. It is in fact a dataset that our supposedly trained model had never seen. 

To the uninitiated, this would be a dataset an end-user would feed to the model to generate a prediction. 
Therefore, it follows that we were expected to manually separate train from test data. 

Another potential consideration that I had to take into account is the shuffling of training data. 
This is an essential step to prevent overfitting as it helps reduce sequential bias and improves generalization.
Another way, I tried to reduce overfitting is through the use of **Random Forests (RF)**. 
RF reduces overfitting by aggregating and averaging out predictions to produce a more realistic result. 
That being said, I wasn't too keen on spending too much time trying to build a model from scratch, and figured that a simple pre-built RF model should suffice. 

I looked up a couple documentations online and found a base code on how I could go about implementing RF in my script.

> https://www.tensorflow.org/decision_forests/tutorials/beginner_colab

A couple things I modified from the base code, other than model creation and evaluation is the train-test split and feature selection. 

I decided to use `pandas` instead of `Tensorflow` to select a subset of features as not only was I  more comfortable with using its functions but also because 
it lets me work with `sklearn` without too much of a hassle to create the train-test split. 
Another benefit of manipulating data this way is that it allows me to separate labels from features before concatenating them together again as distinct 
train-test datasets, whereas the basecode provided by Tensorflow's docs would have you write a function within the script which just makes the script more complicated, messy,
and harder to debug.

In addition, I also neglected to convert the labels to integers as suggested by the documentation, as they already came in binary form, so that 
saved me quite a bit of work.

With that, I trained and evaluated the model where I finally got an accuracy score of 0.8619. 
I was quite pleased with the results and moved on to saving the model, and making a prediction with it given the test data. 
Again, I suppose I could experiment with using XGBoost to increase accuracy, but I'm quite the sloth, if I'm being perfectly honest.
The results are then stored to a `pandas` DataFrame before being saved as .csv under `./prediction`

## Results

I turned my prediction in and received a public score of 0.76794 (the closer it is to 1.0 the better), but I'm not complaining, given the lack of effort in refining
the model. 

Additional info can be found in the comments I wrote within `main.py`

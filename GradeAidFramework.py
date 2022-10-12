import pandas as pd
import nltk
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sentence_transformers.cross_encoder import CrossEncoder

def sim_computation(mypathdataset,pathforpickledump):
    '''
    this function computes the similarity score between students' and reference answers

    :mypathdataset: path to the dataset
    :pathforpickledump: output' path

    it stores the similarity column in a dedicated pickle object
    '''
    model = CrossEncoder('cross-encoder/stsb-roberta-large') #load BERT cross-encoder
    maindf = pd.read_excel(mypathdataset) #path to the dataset
    embeddings_real = maindf['real_answer']
    embeddings_answer = maindf['answer']
    pairs = list([list(x) for x in zip(embeddings_real, embeddings_answer)])
    scores = model.predict(pairs)
    pickle.dump(scores, open(pathforpickledump+".pickle", 'wb')) #save similarity column, add correct path



def jointhem(x1, x2):
    '''
    concatenation between dataset (students, ref) and (similarity score column)

    :param x1: dataset with students' and reference answers
    :param x2: similarity column computed through sim_computation() function
    :return: dataset with students' and reference answers concatenated to the similarity between the two for each pair
    '''
    x2=np.array([x2])
    temp = np.concatenate((x1, x2.T), axis=1)
    return temp


nltk.download('punkt')
mypathdataset="" #provide a path to a dataset
'''
the dataset must comply to the following guidelines
ID,question,reference_ans,student_ans,grade
1,abc?,ccc,aaa,0
1,abc?,ccc,cca,2
...
2,xyz?,www,cww,3
'''
pathoutput="" #provide path output for the experiments
pathforpickledump="" #provide path for dump a pickle of similarities
sim_computation(mypathdataset,pathforpickledump)
sim = pickle.load(open(pathforpickledump+".pickle","rb")) #load similarities

maindf = pd.read_excel(mypathdataset)
y = maindf['grade'] #scores

mystopwords="" #path to stopwords file

with open(mystopwords, 'r', encoding='utf-8') as f:
    engStopwords = set([x[:-1] for x in f.readlines()])

corpus = []
#clean the corpus of answers
for answer in maindf['answer']:
    words = nltk.tokenize.word_tokenize(answer, language='english', preserve_line=False)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in engStopwords]
    sentenceClean = " ".join(words)
    corpus.append(sentenceClean)


kf = KFold()

ids=maindf['ID'] #get the ids of questions
unique=np.unique(ids) #find the number of questions
question={}
for i in range(1,len(unique)+1):
    question['q'+str(i)]=dict() #create a dictionary of questions' datasets
    u=unique[i]
    for (id,j,k,l) in zip(ids,corpus,y,sim): #fill the dictionary of questions' datasets
        if id==u:
            question['q'+str(i)]['corpus']=[]
            question['q'+str(i)]['corpus'].append(j)
            question['q' + str(i)]['y'] = []
            question['q' + str(i)]['y'].append(k)
            question['q' + str(i)]['x'] = []
            question['q' + str(i)]['x'].append(l)
        else:
            break



vectorizer = TfidfVectorizer(min_df=0.03)

'''
Here the user should select the parameters to tune
and the ML methods to use (don't forget to import them)
In the following a couple of examples
'''


#Adaboost
ada_estimators = [311, 1011, 3011]
learning_rate = [0.01, 0.1, 1.0]
loss = ['linear', 'square', 'exponential']

#SVR
C = [0.1, 1, 10, 100]
gamma = np.logspace(-3, 3, 5)
kernel = ['rbf', 'sigmoid']


#AdaBoost
for q in question:
    vector_mae = []
    vector_mse = []
    vector_rmse = []
    max = 1000000
    config = ""
    for estimators in ada_estimators:
        for learning in learning_rate:
            for l in loss:
                vector_mae = []
                vector_mse = []
                vector_rmse = []
                for train_index, test_index in kf.split(question[q]['corpus']):
                    fitting_corpus = [question[q]['corpus'][i] for i in train_index]
                    valid_corpus = [question[q]['corpus'][i] for i in test_index]
                    train_corpus = vectorizer.fit_transform(fitting_corpus).todense()  # training
                    test_corpus = vectorizer.transform(valid_corpus).todense()  # testing
                    fitting_sim = [question[q]['x'][i] for i in train_index]
                    valid_sim = [question[q]['x'][i] for i in test_index]
                    train = jointhem(train_corpus, fitting_sim)
                    test = jointhem(test_corpus, valid_sim)
                    train_labels = [question[q]['y'][i] for i in train_index]
                    test_labels = [question[q]['y'][i] for i in test_index]
                    regress = AdaBoostRegressor(n_estimators=estimators, learning_rate=learning, loss=l)
                    regress.fit(train, train_labels)
                    y_pred = regress.predict(test)
                    mae = mean_absolute_error(test_labels, y_pred)
                    mse = mean_squared_error(test_labels, y_pred, squared=True)
                    rmse = mean_squared_error(test_labels, y_pred, squared=False)
                    vector_mae.append(mae)
                    vector_mse.append(mse)
                    vector_rmse.append(rmse)
                # LOO-CV terminated
                avg_mae = np.mean(vector_mae)
                std_mae = np.std(vector_mae)
                avg_mse = np.mean(vector_mse)
                std_mse = np.std(vector_mse)
                avg_rmse = np.mean(vector_rmse)
                std_rmse = np.std(vector_rmse)
                with open(pathoutput+"AdaBoost_" + q + "_" + str(
                        estimators) + "_" + str(learning) + "_"+l+".txt", "w") as f:
                    f.write("MAE=" + str(avg_mae) + ", std.err=" + str(std_mae))
                    f.write("MSE=" + str(avg_mse) + ", std.err=" + str(std_mse))
                    f.write("RMSE=" + str(avg_rmse) + ", std.err=" + str(std_rmse))
                if max > abs(avg_mse):  # keep trace of best results
                    max = abs(avg_mse)
                    config = "estimators=" + str(estimators) + " learning_rate=" + str(learning)+ " loss="+l

    with open(pathoutput+"AdaBoost_" + q + "_best.txt",
              "w") as f:  # save the best
        f.write("MSE= " + str(max) + " " + config)


#SVR
for q in question:
    vector_mae = []
    vector_mse = []
    vector_rmse = []
    max = 1000000
    config = ""
    for c in C:
        for g in gamma:
            for k in kernel:
                vector_mae = []
                vector_mse = []
                vector_rmse = []
                for train_index, test_index in kf.split(question[q]['corpus']):
                    fitting_corpus = [question[q]['corpus'][i] for i in train_index]
                    valid_corpus = [question[q]['corpus'][i] for i in test_index]
                    train_corpus = vectorizer.fit_transform(fitting_corpus).todense()  # training
                    test_corpus = vectorizer.transform(valid_corpus).todense()  # testing
                    fitting_sim = [question[q]['x'][i] for i in train_index]
                    valid_sim = [question[q]['x'][i] for i in test_index]
                    train = jointhem(train_corpus, fitting_sim)
                    test = jointhem(test_corpus, valid_sim)
                    train_labels = [question[q]['y'][i] for i in train_index]
                    test_labels = [question[q]['y'][i] for i in test_index]
                    regress = SVR(C=c, gamma=g, kernel=k)
                    regress.fit(train, train_labels)
                    y_pred = regress.predict(test)
                    mae = mean_absolute_error(test_labels, y_pred)
                    mse = mean_squared_error(test_labels, y_pred, squared=True)
                    rmse = mean_squared_error(test_labels, y_pred, squared=False)
                    vector_mae.append(mae)
                    vector_mse.append(mse)
                    vector_rmse.append(rmse)
                # LOO-CV terminated
                avg_mae = np.mean(vector_mae)
                std_mae = np.std(vector_mae)
                avg_mse = np.mean(vector_mse)
                std_mse = np.std(vector_mse)
                avg_rmse = np.mean(vector_rmse)
                std_rmse = np.std(vector_rmse)
                with open(pathoutput+"SVR_" + q + "_" + str(
                        estimators) + "_" + str(learning) + "_"+l+".txt", "w") as f:
                    f.write("MAE=" + str(avg_mae) + ", std.err=" + str(std_mae))
                    f.write("MSE=" + str(avg_mse) + ", std.err=" + str(std_mse))
                    f.write("RMSE=" + str(avg_rmse) + ", std.err=" + str(std_rmse))
                if max > abs(avg_mse):  # keep trace of best results
                    max = abs(avg_mse)
                    config = "estimators=" + str(estimators) + " learning_rate=" + str(learning)+ " loss="+l

    with open(pathoutput+"SVR_" + q + "_best.txt",
              "w") as f:  # save the best
        f.write("MSE= " + str(max) + " " + config)

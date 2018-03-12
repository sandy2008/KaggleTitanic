import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras import backend as K


# PassengerId -- A numerical id assigned to each passenger.
# Survived -- Whether the passenger survived (1), or didn't (0). We'll be making predictions for this column.
# Pclass -- The class the passenger was in -- first class (1), second class (2), or third class (3).
# Name -- the name of the passenger.
# Sex -- The gender of the passenger -- male or female.
# Age -- The age of the passenger. Fractional.
# SibSp -- The number of siblings and spouses the passenger had on board.
# Parch -- The number of parents and children the passenger had on board.
# Ticket -- The ticket number of the passenger.
# Fare -- How much the passenger paid for the ticker.
# Cabin -- Which cabin the passenger was in.
# Embarked -- Where the passenger boarded the Titanic.

seed = 42
np.random.seed(seed)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

all_data = pd.concat([train, test])


class DataDigest:

    def __init__(self):
        self.ages = None
        self.fares = None
        self.titles = None
        self.cabins = None
        self.families = None
        self.tickets = None


def get_title(name):
    if pd.isnull(name):
        return "Null"

    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1).lower()
    else:
        return "None"


def get_family(row):
    last_name = row["Name"].split(",")[0]
    if last_name:
        family_size = 1 + row["Parch"] + row["SibSp"]
        if family_size > 3:
            return "{0}_{1}".format(last_name.lower(), family_size)
        else:
            return "nofamily"
    else:
        return "unknown"


def get_index(item, index):
    if pd.isnull(item):
        return -1

    try:
        return index.get_loc(item)
    except KeyError:
        return -1


def data_prepare(data, digest):

    genders = {"male": 1, "female": 0}
    data['SexF'] = data['Sex'].apply(lambda x: genders.get(x))
    gender_dummies = pd.get_dummies(data["Sex"], prefix="SexD", dummy_na=False)
    data = pd.concat([data, gender_dummies], axis=1)

    data['AgeF'] = data.apply(lambda x: digest.ages[x['Sex']] if pd.isnull(x["Age"]) else x["Age"], axis=1)

    data["FareF"] = data.apply(lambda x: digest.fares[x["Pclass"]] if pd.isnull(x["Fare"]) else x["Fare"], axis=1)

    embarkments = {"U": 0, "S": 1, "C": 2, "Q": 3}
    data["EmbarkedF"] = data["Embarked"].fillna("U").apply(lambda x: embarkments.get(x))
    embarkment_dummies = pd.get_dummies(data["Embarked"], prefix="EmbarkedD", dummy_na=False)
    data = pd.concat([data, embarkment_dummies], axis=1)

    data["RelativesF"] = data["Parch"] + data["SibSp"]
    data["SingleF"] = data["RelativesF"].apply(lambda x: 1 if x == 0 else 0)

    decks = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    data["DeckF"] = data["Cabin"].fillna("U").apply(lambda x: decks.get(x[0], -1))

    deck_dummies = pd.get_dummies(data["Cabin"].fillna("U").apply(lambda x: x[0]), prefix="DeckD", dummy_na=False)
    data = pd.concat([data, deck_dummies], axis=1)

    title_dummies = pd.get_dummies(data["Name"].apply(lambda x: get_title(x)), prefix="TitleD", dummy_na=False)
    data = pd.concat([data, title_dummies], axis=1)

    data["CabinF"] = data["Cabin"].fillna("unknown").apply(lambda x: get_index(x, digest.cabins))

    data["TitleF"] = data["Name"].apply(lambda x: get_index(get_title(x), digest.titles))

    data["TicketF"] = data["Ticket"].apply(lambda x: get_index(x, digest.tickets))

    data["FamilyF"] = data.apply(lambda x: get_index(get_family(x), digest.families), axis=1)

    return data


data_digest = DataDigest()
data_digest.ages = all_data.groupby("Sex")["Age"].median()
data_digest.fares = all_data.groupby("Pclass")["Fare"].median()

titles_trn = pd.Index(train["Name"].apply(get_title).unique())
titles_tst = pd.Index(test["Name"].apply(get_title).unique())
data_digest.titles = titles_tst

families_trn = pd.Index(train.apply(get_family, axis=1).unique())
families_tst = pd.Index(test.apply(get_family, axis=1).unique())
data_digest.families = families_tst

cabins_trn = pd.Index(train["Cabin"].fillna("unknown").unique())
cabins_tst = pd.Index(test["Cabin"].fillna("unknown").unique())
data_digest.cabins = cabins_tst

tickets_trn = pd.Index(train["Ticket"].fillna("unknown").unique())
tickets_tst = pd.Index(test["Ticket"].fillna("unknown").unique())
data_digest.tickets = tickets_tst

# List of feature for selection
predictors = ["Pclass",
              "AgeF",
              "TitleF",
              "TitleD_mr", "TitleD_mrs", "TitleD_miss", "TitleD_master", "TitleD_ms",
              "TitleD_col", "TitleD_rev", "TitleD_dr",
              "CabinF",
              "DeckF",
              "DeckD_U",
              "DeckD_A",
              "DeckD_B", "DeckD_C", "DeckD_D", "DeckD_E", "DeckD_F",
              "DeckD_G",
              "FamilyF",
              "TicketF",
              "SexF",
              "SexD_male", "SexD_female",
              "EmbarkedF",
              "EmbarkedD_S", "EmbarkedD_C",
              "EmbarkedD_Q",
              "FareF",
              "SibSp",
              "Parch",
              "RelativesF",
              'SingleF',
              ]

pre_train = data_prepare(train, data_digest)
pre_test = data_prepare(test, data_digest)
pre_all = pd.concat([pre_train, pre_test])

scaler = StandardScaler()
scaler.fit(pre_all[predictors])

train_data_scaled = scaler.transform(pre_train[predictors])
test_data_scaled = scaler.transform(pre_test[predictors])

selector = SelectKBest(f_classif, k=5)
selector.fit(pre_train[predictors], pre_train["Survived"])

scores = -np.log10(selector.pvalues_)

# Show predictors
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

X = pre_train[predictors].as_matrix()
Y = np_utils.to_categorical(pre_train['Survived'])
Y = Y[:, 1]

kfold = StratifiedKFold(n_splits=5,  shuffle=True, random_state=seed).split(X, Y)
cv = []
i = len(predictors)
index = test["PassengerId"].as_matrix()

def load_model_and_fit(train_x, train_y):
    model = Sequential()
    model.add(Dense(32, input_dim=i, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(train_x, train_y, nb_epoch=600, batch_size=32, verbose=0)
    return model

# For testing
for train, test in kfold:
    model = load_model_and_fit(X[train], Y[train])
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cv.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv), np.std(cv)))


model = load_model_and_fit(X, Y)
out = model.predict_classes(pre_test[predictors].as_matrix()).ravel()


submission = pd.DataFrame({
        "PassengerId": index,
        "Survived": out
})
filename = 'titanic-keras.csv'

submission.to_csv(filename, index=False)
print('\n\n %s save' % filename)

K.clear_session()
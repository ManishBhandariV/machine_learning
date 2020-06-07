import pandas as pd
import numpy as np
import sklearn
import nltk
import os
import urllib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import  svm


cwd = os.getcwd()

datapoints = pd.read_csv(r"master996.csv",sep=';',engine='python')

#print(datapoints.columns)
#print(datapoints["guten_genre"].unique())
# x = datapoints[["Book_Name", "Author_Name"]]
# y = datapoints["guten_genre"]


columns = ['Book_Name', 'book_id', 'guten_genre', 'Author_Name']
# d_m_lst= []
# lit_lst = []
# w_s_lst = []
# g_h_lst = []
# cs_lst = []
# l_r_lst = []
# s_a_lst = []
# alle_lst = []
# h_w_s_lst = []
#
#
#
#
# for i in range(len(datapoints)):
#     if (datapoints["guten_genre"][i] == "Detective and Mystery"):
#         d_m_lst.append(datapoints.iloc[i,:])
#     elif (datapoints["guten_genre"][i] == "Literary"):
#         lit_lst.append(datapoints.iloc[i,:])
#     elif (datapoints["guten_genre"][i] == "Western Stories"):
#         w_s_lst.append(datapoints.iloc[i,:])
#     elif (datapoints["guten_genre"][i] == "Ghost and Horror"):
#         g_h_lst.append(datapoints.iloc[i,:])
#     elif (datapoints["guten_genre"][i] == "Christmas Stories"):
#         cs_lst.append(datapoints.iloc[i,:])
#     elif (datapoints["guten_genre"][i] == "Love and Romance"):
#         l_r_lst.append(datapoints.iloc[i,:])
#     elif (datapoints["guten_genre"][i] == "Sea and Adventure"):
#         s_a_lst.append(datapoints.iloc[i,:])
#     elif (datapoints["guten_genre"][i] == "Allegories"):
#         alle_lst.append(datapoints.iloc[i,:])
#     elif (datapoints["guten_genre"][i] == "Humorous and Wit and Satire"):
#         h_w_s_lst.append(datapoints.iloc[i,:])
#
#
#
#
#
#
# dm = pd.DataFrame(data= d_m_lst, columns= columns)
# lit = pd.DataFrame(data= lit_lst, columns= columns)
# ws = pd.DataFrame(data= w_s_lst, columns= columns)
# gh = pd.DataFrame(data= g_h_lst, columns= columns)
# cs = pd.DataFrame(data= cs_lst, columns= columns)
# lr = pd.DataFrame(data= l_r_lst, columns= columns)
# sa = pd.DataFrame(data= s_a_lst, columns= columns)
# ale = pd.DataFrame(data= alle_lst, columns= columns)
# hws = pd.DataFrame(data= h_w_s_lst, columns= columns)
#
# seperate_list = [dm, lit, ws, gh, cs, lr, sa, ale, hws]
# # for i in seperate_list:
# #     print(len(i))
# sample_list = []
# for i in range(len(seperate_list)):
#     if len(seperate_list[i]) > 10:
#         lcl = seperate_list[i].sample(n = 10)
#         for i in range(len(lcl)):
#             sample_list.append(lcl.iloc[i,:])
#     else:
#         lce = seperate_list[i].sample(n=len(seperate_list[i]))
#         for i in range(len(lce)):
#             sample_list.append(lce.iloc[i,:])
#
# sample = pd.DataFrame(data= sample_list, columns= columns)
rowcount = len(datapoints.axes[0])
data = []
dataclass = []
for i in range(0, rowcount):
    bookid = datapoints.iloc[i, 1]
    bookid_split = bookid.split('.')
    dataclass.append(datapoints.iloc[i, 2])
    path = "file:///" + cwd + "\Gutenberg_19th_century_English_Fiction/" + bookid_split[0] + "-content.html"
    data.append(urllib.request.urlopen(path).read())

tags_index = {'Detective and Mystery': 1, 'Literary': 2, 'Western Stories': 3, 'Ghost and Horror': 4,
              'Christmas Stories': 5, 'Love and Romance': 6, 'Sea and Adventure': 7, 'Allegories': 8,
              'Humorous and Wit and Satire': 9}

# print(data[0])
# print(dataclass[1])


def preprocessing(String, stopwordFlag=True,
                  stemmingFlag= True):  # default value is always true for stemming and stopwords

    '''
    This function is used for preprocessing
    - Tokenization
    - Stemming
    - Stop Words

    '''

    tokens = nltk.word_tokenize(String)
    token = [word for word in tokens if word.isalpha()]
    #   token = [token.remove(word) for word in token if (word.isalpha())== False]

    if stopwordFlag == False and stemmingFlag == False:
        token_string = " ".join(token)
        return token_string

    stop_words = set(stopwords.words('english'))
    influentialwords = []
    stemwords = []
    for w in token:
        if w not in stop_words:
            influentialwords.append(w)
    if stemmingFlag == False:
        stopwords_string = " ".join(stemwords)
        return stopwords_string

    ps = PorterStemmer()

    if stopwordFlag == False:
        for w in token:
            stemwords.append(ps.stem(w))
        stemwords_string = " ".join(stemwords)
        return stemwords_string
    for w in influentialwords:
        stemwords.append(ps.stem(w))
    stemwords_string = " ".join(stemwords)
    return stemwords_string


data_preprocess = []

for i in range(0, rowcount):
    data_preprocess.append(preprocessing(str(data[i])))

for i in range(0, rowcount):
    data_preprocess[i] = ' '.join([w for w in str(data_preprocess[i]).split() if len(w) > 1])

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data_preprocess)
print(vectors.shape)
# print(vectorizer.get_feature_names())
# print(newsgroup_train.filenames.shape)
feature_names = np.array(vectorizer.get_feature_names())
# test_vect = vectorizer.transform(data_preprocess[0])
# chk = [data_preprocess[0]]
# test_vect = vectorizer.transform(chk)
def get_top_tf_idf_words(response, top_n=1000):
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    return feature_names[response.indices[sorted_nzs]]
#print([get_top_tf_idf_words(response,1000) for response in test_vect])

detmys_wrd = []
lit_wrd = []
ws_wrd = []
gh_wrd = []
cs_wrd = []
lr_wrd = []
sa_wrd = []
al_wrd = []
hws_wrd = []

for i in range(0,len(data_preprocess)):

    if dataclass[i] == "Detective and Mystery":
        pre_vect = [data_preprocess[i]]
        vect = vectorizer.transform(pre_vect)
        detmys_wrd.append([get_top_tf_idf_words(response,1000) for response in vect])
    elif dataclass[i] == "Literary":
        pre_vect = [data_preprocess[i]]
        vect = vectorizer.transform(pre_vect)
        lit_wrd.append([get_top_tf_idf_words(response,1000) for response in vect])
    elif dataclass[i] == "Western Stories":
        pre_vect = [data_preprocess[i]]
        vect = vectorizer.transform(pre_vect)
        ws_wrd.append([get_top_tf_idf_words(response,1000) for response in vect])
    elif dataclass[i] == "Ghost and Horror":
        pre_vect = [data_preprocess[i]]
        vect = vectorizer.transform(pre_vect)
        gh_wrd.append([get_top_tf_idf_words(response,1000) for response in vect])
    elif dataclass[i] == "Christmas Stories":
        pre_vect = [data_preprocess[i]]
        vect = vectorizer.transform(pre_vect)
        cs_wrd.append([get_top_tf_idf_words(response,1000) for response in vect])
    elif dataclass[i] == "Love and Romance":
        pre_vect = [data_preprocess[i]]
        vect = vectorizer.transform(pre_vect)
        lr_wrd.append([get_top_tf_idf_words(response,1000) for response in vect])
    elif dataclass[i] == "Sea and Adventure":
        pre_vect = [data_preprocess[i]]
        vect = vectorizer.transform(pre_vect)
        sa_wrd.append([get_top_tf_idf_words(response,1000) for response in vect])
    elif dataclass[i] == "Allegories":
        pre_vect = [data_preprocess[i]]
        vect = vectorizer.transform(pre_vect)
        al_wrd.append([get_top_tf_idf_words(response,1000) for response in vect])
    else:
        pre_vect = [data_preprocess[i]]
        vect = vectorizer.transform(pre_vect)
        hws_wrd.append([get_top_tf_idf_words(response, 1000) for response in vect])

detmys_wrds = []
lit_wrds = []
ws_wrds = []
gh_wrds = []
cs_wrds = []
lr_wrds = []
sa_wrds = []
al_wrds = []
hws_wrds = []
for i in range(len(detmys_wrd)):
    for j in range(len(detmys_wrd[i])):
        for k in range(len(detmys_wrd[i][j])):
            detmys_wrds.append(detmys_wrd[i][j][k])

for i in range(len(lit_wrd)):
    for j in range(len(lit_wrd[i])):
        for k in range(len(lit_wrd[i][j])):
            lit_wrds.append(lit_wrd[i][j][k])
for i in range(len(ws_wrd)):
    for j in range(len(ws_wrd[i])):
        for k in range(len(ws_wrd[i][j])):
            ws_wrds.append(ws_wrd[i][j][k])
for i in range(len(gh_wrd)):
    for j in range(len(gh_wrd[i])):
        for k in range(len(gh_wrd[i][j])):
            gh_wrds.append(gh_wrd[i][j][k])
for i in range(len(cs_wrd)):
    for j in range(len(cs_wrd[i])):
        for k in range(len(cs_wrd[i][j])):
                cs_wrds.append(cs_wrd[i][j][k])
for i in range(len(lr_wrd)):
    for j in range(len(lr_wrd[i])):
        for k in range(len(lr_wrd[i][j])):
            lr_wrds.append(lr_wrd[i][j][k])

for i in range(len(sa_wrd)):
    for j in range(len(sa_wrd[i])):
        for k in range(len(sa_wrd[i][j])):
            sa_wrds.append(sa_wrd[i][j][k])

for i in range(len(al_wrd)):
    for j in range(len(al_wrd[i])):
        for k in range(len(al_wrd[i][j])):
            al_wrds.append(al_wrd[i][j][k])

for i in range(len(hws_wrd)):
    for j in range(len(hws_wrd[i])):
        for k in range(len(hws_wrd[i][j])):
            hws_wrds.append(hws_wrd[i][j][k])

#add custom words from net to each list
line_det_mys_list = []
line_det_mys= " Abduction, Abuse, Access, Accident, Accuse, Action, Admission, Adult, Agency, Agree, Alarm, Alert, Alias, Allege, Appeal, Appearance, Appraise, Archives, Armed, Arraignment, Arrest, Arson, Ask, Aspect, Assignment, Assistance, Assumptions, Attitude, Authenticate, Authority, Authorize, Backup, Badge, Ballistics, Basis, Battery, Behavior, Belief, Blackmail, Bloodstain, Bodyguard, Bomb squad, Bond, Booking, Branch, Breach, Bribes, Brutal, Brutality, Burden, Bureau, Burglary, Busted, By-the-book, Capable, Captain, Capture, Careful, Catch, Cautious, Cease, Challenges, Character, Chase, Check out, Citation, Citizen, Civil, Claim, Code, Cold case, Colleague, Collude, Collusion, Commission, Commit, Communication, Community, Competitive, Complaints, Complicated, Concerned, Conduct, Confer, Confess, Confession, Confidential, Confrontation, Consent, Consider, Conspiracy, Consult, Contempt, Convict, Conviction, Cooperate, Cop, Coroner, Corrupt, Counterfeit, Court, Crimes, Criminal, Cruise, Damage, Danger, Dangerous, Dealings, Decisions, Dedication, Deduction, Deed, Deed, Defense, Deliberate, Delinquent, Deliver, Denial, Deny, Department, Deputy, Detain, Detect, Detective, Determination, Deviant, Dialogue, Difficult, Direct, Disappearance, Discovery, Disobedient, Disorderly, Dispatch, Disregard, District attorney, Documentation, Documents, Domestic disputes, Doubtful, Drugs, Drunk, Dupe, Duty, Dying, Educate, Education, Effect, Embezzle, Emergency, Emphasis, Enable, Encounter, Encumber, Enforce, Entail, Entrap, Equality, Equipment, Espionage, Ethical, Evidence, Examine, Execute, Experience, Expert, Expose, Extort, Extradition, Extreme, Eyes, Facts, Failure, Fairness, Family, FBI, Federal, Feisty, Felony, Fight, File, Fine, Fingerprint, Follow, Follow-up, Footprints, Force, Forgery, Formal charges, Foul play, Fraud, Freedom, Full-scale, Fundamental, Gang, Gore, Government, Guarantee, Guard, Guilty, Gum shoe, Gun, Handcuff, Handle, Harmful, Helpful, High-powered, Hijack, Hire, Holding, Homicide, Honest, Honor, Hostage, Ill-gotten, Illegal, Illegitimate, Immoral, Imprison, Inappropriate, Incompetent, Indict, Influence, Informant, Information, Initiative, Injury, Innocent, Innuendo, Inquest, Inquire, Instinct, Intelligence, Interests, Interfere, Internet, Interpol, Interpretation, Interstate, Intuition, Investigate, Investigation, Irregular, Issue, Jail, John Doe, Judge, Judgment, Judicial, Jury, Justice, Juvenile, Juvvy, Kidnapping, Kin, Knowledge, Laboratory, Larceny, Law, Lawful, Lawsuit, Lease, Legacy, Legal, Legitimate, Libel, Liberty, Licensed, Lie, Lieutenant, Limit, Links, Long hours, Lurk, Magistrate, Maintain, Majority, Malevolence, Malicious, Manslaughter, Mayhem, Menace, Minority, Miscreant, Misdemeanor, Missing person, Mission, Mob, Motivation, Motive, Motor pool, Motorist, Murder, Mystery, National, Negligence, Negotiate, Negotiate, Neighborhood, Notation, Notification, Nuisance, Oath, Obey, Obligation, Obsession, Offender, Offense, Officer, Official, Omission, Opinion, Opportunity, Order, Organize, Paper work,Parole, Partner, Partnership, Patrol, Patterns, Pedestrian, Penalize, Penalty, Penitentiary, Penny-ante, Perjury, Perpetrator, Phony, Plain-clothes officer, Plead, Police, Police academy, Power, Precedent, Prevention, Previous, Principle, Priors, Prison, Private, Probable cause, Probation officer, Procedure, Process, Professional, Profile, Proof, Property, Prosecutor, Protection, Prove, Provision, Public, Punishment, Qualification, Quality, Quantify, Quantity, Quarrel, Quell, Query, Question, Quick, Quirks, Radar, Rank, Reading rights, Reasons, Record, Recruit, Red-handed, Redemption, Redress, Reduction, Refute, Register, Registration, Regulation, Reinforcements, Reject, Release, Report, Reports, Reprobate, Reputation, Research, Resist, Response, Responsibility, Restraining order, Restrict, Retainer, Revenge, Rights, Riot, Robbery, Rogue, Routine, Rules, Rulings, Sabotage, Safeguard, Safety, Sanction, Scandal, Scene, Scum bag, Sealed record, Search and rescue team, Searching, Secret, Seize, Select, Sentence, Sergeant, Seriousness, Serve, Services, Sheriff, Shift, Shooting, Shyster, Sighting, Situation, Skilled, Slander, Slaying, Sleazy, Sleuthing, Smuggling, Snitch, Solution, Solve, Sources, Squad, Stalk, State, Statute, Statute of limitation, Stipulation, Strangulation, Study, Subdue, Subpoena, Successful, Sully, Summons, Suppression, Surveillance, Suspect, Suspected, Suspicion, Suspicious, Sworn, System, Tactics, Tantamount, Taping, Tazer, Technique, Tense, Tension, Testify, Testimony, Theory, Threatening, Thwart, Tip, Traffic, Transfer, Trap, Treatment, Trespass, Trial, Trooper, Trust, Truth, Unacceptable, Unauthorized, Unclaimed, Unconstitutional, Undercover, Underpaid, Unintentional, Unit, Unjust, Unknown, Unlawful, Uphold, Urgency, Vagrant, Vandalism, Vanish, Verdict, Verification, Victim, Victimize, Viewpoint, Vigilante, Villain, Violate, Violation, Violence, Volume, Warped, Warrant, Watch, Weapon, Whodunit, Will, Wiretap, Wisdom, Witness, Wrong, Young, Youth, Zap, Zeal, Zealous, Zero, Zilch"
line_det_mys_list.append(line_det_mys.split())
for i in range(len(line_det_mys_list[0])):
    detmys_wrds.append(line_det_mys_list[0][i])


line_lit_list = []
line_lit= "allegory, alliteration, allusion, anachronism, anaclisis, anadiplosis, analogy, anaphora, anastrophe, anecdote, antagonist, anthropomorphism, antithesis, antonym, apogee, aposiopesis, apostrophe, aside, assonance, asyndeton, auditory, ballad, bard, bathos, beholden, blank verse, bombast, canticle, carter, catachresis, catechism, catharsis, chiasmus, chicanery, collocation, colloquialism, confound, consonance, couplet, cryptic, curtal, deixis, demotic, Demotic script, denouement, derogatory, dialogue, diction, didactic, dissemble, dolour, drama, dramatic irony, dug, elegy, ellipsis, emblem, end-stopped, enjambment, enmity, epic poetry, epigram, epiphany, epistolary, epistrophe, epitaph, epitome, epoch, equanimity, ethos, eulogy, euphemism, explicit, fable, fief, figurative, flout, foreshadow, forsake, free verse, frisson, guile, hearse, heroic couplet, homonym, humility, hymn, hyperbaton, hyperbole, hypozeuxis, iambic, imagery, implicit, importune, incensed, inversion, invidious, irony, kinesthetic, lamentation, liege, listing, litotes, Logos, lyricism, malapropism, malignant, meiosis, meme, metaphor, metonymy, mimicry, moiety, motif, nadir, narrative, neologism, ode, olfactory, omniscient, onomatopoeia, otiose, oxymoron, paean, parable, paradox, paradoxical sleep, paragon, parallelism, parallelism, parody, paronomasia, pastoral, pathos, pejorative, persnickety, personification, ploce, poetry, polyptoton, polysyndeton, portmanteau word, preface, premonitory, prologue, prose, prosody, protagonist, pun, purge, rancour, redeemer, refrain, repetition, requiem, sardonic, satire, simile, soliloquy, sonnet, sophistry, specious argument, speech rhythm, sprung rhythm, stanza, surrealism, symbolism, symploce, synaesthesia, synecdoche, synesthetic metaphor, synonym, syntax, tactile, tautology, theme, tone, topos, totem, trope, unwittingly, vassal, verisimilitude, versification, visual, wretched, zeal, zeugma, terza rima, heroic couplet, apostrophe, assonance, consonance, onomatopoeia, novelette, novella, epic poem, foible, mock-heroic, verisimilitude, third person, first person, second person, omniscient, omnipresent, fable, bestial, apposition, anaphora, tactile, auditory, visual, olfactory, epithalamium, anapest, neologism, heath, fern, portend, conceit, unrequited, epithet, dystopia, utopia, leitmotif, moral, motif, semiotics, fable, burlesque, trope, tanka, haiku, ode, strophe, anastrophe, acrostic, anachronistic, apothegm, autotelism, avant-garde, baroque"
line_lit_list.append(line_lit.split())
for i in range(len(line_lit_list[0])):
    lit_wrds.append(line_lit_list[0][i])
print('done')

tag_data =  [TaggedDocument(d.split(), [i]) for i, d in enumerate(data_preprocess)]

glb_lst = []
for i in range(len(tag_data)):
    lcl_lst = []
    lcl_lst.clear()
    lcl_lst.append(round(len(set(tag_data[i][0])& set(detmys_wrds))/len(detmys_wrds),2))
    lcl_lst.append(round(len(set(tag_data[i][0])& set(lit_wrds))/len(lit_wrds),2))
    lcl_lst.append(round(len(set(tag_data[i][0]) & set(ws_wrds)) / len(ws_wrds),2))
    lcl_lst.append(round(len(set(tag_data[i][0]) & set(gh_wrds)) / len(gh_wrds),2))
    lcl_lst.append(round(len(set(tag_data[i][0]) & set(cs_wrds)) / len(cs_wrds),2))
    lcl_lst.append(round(len(set(tag_data[i][0]) & set(lr_wrds)) / len(lr_wrds),2))
    lcl_lst.append(round(len(set(tag_data[i][0]) & set(sa_wrds)) / len(sa_wrds),2))
    lcl_lst.append(round(len(set(tag_data[i][0]) & set(al_wrds)) / len(al_wrds),2))
    lcl_lst.append(round(len(set(tag_data[i][0]) & set(hws_wrds)) / len(hws_wrds),2))
    glb_lst.append(lcl_lst)

print('done')

df = pd.DataFrame(data= glb_lst, columns=["detmys_wrds","lit_wrds","ws_wrds","gh_wrds","cs_wrds","lr_wrds","sa_wrds","al_wrds","hws_wrds"])
df["class"] = dataclass

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1], test_size= 0.2,random_state= 42)

# classifier_knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
# classifier_knn.fit(x_train,y_train)
# y_pred = classifier_knn.predict(x_test)
# print(accuracy_score(y_train, classifier_knn.predict(x_train)))
# print(accuracy_score(y_test, y_pred))
# #print(confusion_matrix(y_test,y_pred,labels= ['Detective and Mystery', 'Literary', 'Western Stories','Ghost and Horror', 'Christmas Stories', 'Love and Romance','Sea and Adventure', 'Allegories', 'Humorous and Wit and Satire']))
#
# classifier_dt = DecisionTreeClassifier( criterion= 'entropy',max_depth= 5)
# classifier_dt.fit(x_train,y_train)
# y_pred = classifier_dt.predict(x_test)
# print(accuracy_score(y_train, classifier_dt.predict(x_train)))
# print(accuracy_score(y_test, y_pred))

classifier_svm = svm.SVC(kernel= 'rbf')
classifier_svm.fit(x_train,y_train)
y_pred = classifier_svm.predict(x_test)
print(accuracy_score(y_train, classifier_svm.predict(x_train)))
print(accuracy_score(y_test, y_pred))


titles_options = [("Confusion matrix, without normalization", None),("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier_svm, x_test, y_test,
                                 display_labels=['Detective and Mystery', 'Literary', 'Western Stories','Ghost and Horror', 'Christmas Stories', 'Love and Romance','Sea and Adventure', 'Allegories', 'Humorous and Wit and Satire'],
                                 cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()




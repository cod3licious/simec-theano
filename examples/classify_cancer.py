from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.decomposition import KernelPCA
import sklearn.metrics as skmet

# https://github.com/cod3licious/nlputils
from nlputils.features import FeatureTransform, features2mat
from simec.similarity_encoder import SimilarityEncoder, scaled_sigmoid

# https://github.com/cod3licious/cancer_papers
from datasets.cancer_papers.load_cancer import articles2dict


def classify_sklearn(X, X_test, y, y_test):
    # classify with SVM
    clf = LinearSVC(class_weight='balanced', random_state=1)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    print("F1 micro-avg: %.3f, F1 macro-avg: %.3f" % (100. * skmet.f1_score(y_test, y_pred, average='micro'),
                                                      100. * skmet.f1_score(y_test, y_pred, average='macro')))


if __name__ == '__main__':
    # load cancer dataset with 2 types of labels
    textdict, doccats_k, catdesc_k = articles2dict(label='keyword', reduced_labels=True,
                                                   combine_paragraphs=False, ignore_types=['Mixed'], verbose=0)
    textdict, doccats_p, catdesc_p = articles2dict(label='partype', reduced_labels=True,
                                                   ignore_types=['Mixed'], verbose=0)
    # split in train and test dataset + get label vectors
    np.random.seed(42)
    doc_ids = list(np.random.permutation(sorted(textdict.keys())))[:10000]
    textdict = {d: textdict[d] for d in doc_ids}
    train_ids = doc_ids[:int(.8 * len(doc_ids))]
    test_ids = doc_ids[int(.8 * len(doc_ids)):]
    y_train_k = np.array([doccats_k[i] for i in train_ids])
    y_train_p = np.array([doccats_p[i] for i in train_ids])
    y_test_k = np.array([doccats_k[i] for i in test_ids])
    y_test_p = np.array([doccats_p[i] for i in test_ids])
    print("[INFO] %i training documents and %i test documents" % (len(train_ids), len(test_ids)))

    # compute length renormed tf-idf features and analyze
    print("[INFO] tf-idf with **length** renorm")
    ft = FeatureTransform(norm='max', weight=True, renorm='length')
    docfeats = ft.texts2features(textdict, train_ids)
    print("[INFO] train matrix...", end=' ')
    X, featurenames = features2mat(docfeats, train_ids)
    print("with %i features ...test matrix..." % len(featurenames))
    X_test, _ = features2mat(docfeats, test_ids, featurenames)
    # classify
    print("[INFO] classify based on **keyword**")
    classify_sklearn(X, X_test, y_train_k, y_test_k)
    print("[INFO] classify based on **partype**")
    classify_sklearn(X, X_test, y_train_p, y_test_p)

    # compute kPCA on these features
    print("[INFO] linear kPCA on the length normed features")
    e_lkpca = KernelPCA(n_components=30, kernel='linear')
    X_embed = e_lkpca.fit_transform(X)
    X_embed_test = e_lkpca.transform(X_test)
    print("[INFO] classify based on **keyword**")
    classify_sklearn(X_embed, X_embed_test, y_train_k, y_test_k)
    print("[INFO] classify based on **partype**")
    classify_sklearn(X_embed, X_embed_test, y_train_p, y_test_p)

    # SimEc 50/50 based on cancer- and partype labels
    print("[INFO] tf-idf with **max** renorm")
    ft = FeatureTransform(norm='max', weight=True, renorm='max')
    docfeats = ft.texts2features(textdict, train_ids)
    print("[INFO] train matrix...", end=' ')
    X, featurenames = features2mat(docfeats, train_ids)
    print("test matrix...")
    X_test, _ = features2mat(docfeats, test_ids, featurenames)
    print("[INFO] computing SimEc features based on keyword+partype labels")
    Y = np.tile(y_train_k, (len(y_train_k), 1))
    S = 0.5 * np.array(Y == Y.T, dtype=float)
    Y = np.tile(y_train_p, (len(y_train_p), 1))
    S += 0.5 * np.array(Y == Y.T, dtype=float)
    e_lsimenc = SimilarityEncoder(S.shape[1], X.shape[1], 30, [300],
                                  [scaled_sigmoid, None, scaled_sigmoid], sparse_features=True,
                                  orthOT_reg=0.00025, lrate=15., lrate_decay=1.)
    e_lsimenc.fit(X, S, max_epochs=500)
    X_embed = e_lsimenc.transform(X)
    X_embed_test = e_lsimenc.transform(X_test)
    print("[INFO] classify based on **keyword**")
    classify_sklearn(X_embed, X_embed_test, y_train_k, y_test_k)
    print("[INFO] classify based on **partype**")
    classify_sklearn(X_embed, X_embed_test, y_train_p, y_test_p)

    # SimEc based on cancer type labels
    print("[INFO] computing SimEc features based on keyword labels")
    Y = np.tile(y_train_k, (len(y_train_k), 1))
    S = np.array(Y == Y.T, dtype=int)
    e_lsimenc = SimilarityEncoder(S.shape[1], X.shape[1], 30, [300],
                                  [scaled_sigmoid, None, scaled_sigmoid], sparse_features=True,
                                  orthOT_reg=0.00025, lrate=15., lrate_decay=1.)
    e_lsimenc.fit(X, S, max_epochs=500)
    X_embed = e_lsimenc.transform(X)
    X_embed_test = e_lsimenc.transform(X_test)
    print("[INFO] classify based on **keyword**")
    classify_sklearn(X_embed, X_embed_test, y_train_k, y_test_k)
    print("[INFO] classify based on **partype**")
    classify_sklearn(X_embed, X_embed_test, y_train_p, y_test_p)

    # SimEc based on partype labels
    print("[INFO] computing SimEc features based on partype labels")
    Y = np.tile(y_train_p, (len(y_train_p), 1))
    S = np.array(Y == Y.T, dtype=int)
    e_lsimenc = SimilarityEncoder(S.shape[1], X.shape[1], 30, [300],
                                  [scaled_sigmoid, None, scaled_sigmoid], sparse_features=True,
                                  orthOT_reg=0.00025, lrate=15., lrate_decay=1.)
    e_lsimenc.fit(X, S, max_epochs=500)
    X_embed = e_lsimenc.transform(X)
    X_embed_test = e_lsimenc.transform(X_test)
    print("[INFO] classify based on **keyword**")
    classify_sklearn(X_embed, X_embed_test, y_train_k, y_test_k)
    print("[INFO] classify based on **partype**")
    classify_sklearn(X_embed, X_embed_test, y_train_p, y_test_p)

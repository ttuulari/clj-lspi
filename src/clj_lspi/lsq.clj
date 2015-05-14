(ns clj-lspi.lsq
  (:refer-clojure :exclude [* - + == /])
  (:require
    [clojure.core.matrix :refer :all]
    [clojure.core.matrix.operators :refer :all]))

(defn feature-matrix
  "Approximate feature matrix Phi based on data state-actions."
  [features training-data]
  (let [extract-feature-data (fn [elem]
                               [(:old-state elem)
                                (:action elem)])
        feature-data         (map extract-feature-data training-data)
        mapper               (apply juxt features)]
    (matrix (map mapper feature-data))))

(defn feature-transition-matrix
  "Approximate feature transition matrix based on policy fn"
  [features training-data policy]
  (let [extract-feature-data (fn [elem]
                               [(:new-state elem)
                                (policy (:new-state elem))])
        feature-data         (map extract-feature-data training-data)
        mapper               (apply juxt features)]
    (matrix (map mapper feature-data))))

(defn extract-rewards
  [training-data]
  (matrix (map :reward training-data)))

(defn weights
  [A b]
  (mmul (inverse A) b))

(defn update-a
  [a-matrix features policy discount sample]
  (let [extract-feature-data  (fn [elem]
                                [(:old-state elem)
                                 (:action elem)])
        extract-policy-data   (fn [elem]
                                [(:new-state elem)
                                 (policy (:new-state elem))])
        feature-data          (extract-feature-data sample)
        feature-policy-data   (extract-policy-data sample)
        feature-values        ((apply juxt features) feature-data)
        feature-policy-values ((apply juxt features) feature-policy-data)
        delta                 (->> feature-policy-values
                                   (* discount)
                                   (- feature-values)
                                   (outer-product feature-values))]
    (+ a-matrix delta)))

(defn update-b
  [b-vec features sample]
  (let [extract-feature-data  (fn [elem]
                                [(:old-state elem)
                                 (:action elem)])
        feature-data          (extract-feature-data sample)
        feature-values        ((apply juxt features) feature-data)]
    (->> feature-values
         (* (:reward sample))
         (+ b-vec))))

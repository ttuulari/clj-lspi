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

(defn- extract-rewards
  "Rewards from training data."
  [training-data]
  (-> (map :reward training-data)
      matrix
      transpose))

(defn- weights
  "Solve weight vector by inversing A."
  [A b]
  (mmul (inverse A) b))

(defn solve-weights
  "Solve the weight vector."
  [features training-data policy discount]
  (let [phi     (feature-matrix features training-data)
        p-phi   (feature-transition-matrix features training-data policy)
        phi-t   (transpose phi)
        b       (mmul phi-t 
                      (extract-rewards training-data))
        a       (mmul phi-t
                      (- phi
                         (* discount
                            p-phi)))]
    (weights a b)))

(defn update-a
  "Update a-matrix with new data sample"
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
  "Update b-vector with new data sample"
  [b-vec features sample]
  (let [extract-feature-data  (fn [elem]
                                [(:old-state elem)
                                 (:action elem)])
        feature-data          (extract-feature-data sample)
        feature-values        ((apply juxt features) feature-data)]
    (->> feature-values
         (* (:reward sample))
         (+ b-vec))))

(defn pick-action
  "Pick the actions among (possible actions) that maximizes the state Q-value"
  [state features weights possible-actions]
  (let [actions               (possible-actions state)
        state-action-data     (map (fn [action] 
                                     {:old-state state
                                      :action action})
                                   actions)

        feature-values        (feature-matrix features state-action-data)
        mult-with-weights     (fn [feat-val]
                                (mmul feat-val weights))
        scores                (map mult-with-weights feature-values)
        values                (map vector scores actions)
        sorted                (sort-by first > values)]
    (-> sorted first second)))


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
  [features training-data policy-fn]
  (let [extract-feature-data (fn [elem]
                               [(:new-state elem)
                                (policy-fn (:new-state elem))])
        feature-data         (map extract-feature-data training-data)
        mapper               (apply juxt features)]
    (matrix (map mapper feature-data))))

(defn- extract-rewards
  "Rewards from training data."
  [training-data]
  (-> (map :reward training-data)
      matrix
      transpose))

(defn weights
  "Solve weight vector by inversing A."
  [A b]
  (mmul (inverse A) b))

(defn a-and-b
  [features training-data policy-fn discount]
  (let [phi     (feature-matrix features training-data)
        p-phi   (feature-transition-matrix features training-data policy-fn)
        phi-t   (transpose phi)
        b       (mmul phi-t 
                      (extract-rewards training-data))
        a       (mmul phi-t
                      (- phi
                         (* discount
                            p-phi)))]
    {:a a :b b}))

(defn solve-weights
  "Solve the weight vector."
  [features training-data policy discount]
  (let [{:keys [a b]}  (a-and-b features training-data policy discount)]
    (weights a b)))

(defn update-a
  "Update a-matrix with new data sample"
  [a-matrix features policy discount sample]
  (let [feature-data          [(:old-state sample) (:action sample)]
        feature-policy-data   [(:new-state sample) (policy (:new-state sample))]
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
  (let [feature-data          [(:old-state sample) (:action sample)]
        feature-values        ((apply juxt features) feature-data)]
    (+ (* (:reward sample)
          feature-values)
       b-vec)))

(defn policy-action
  "Pick the action among (possible actions) that maximizes the state argument Q-value.
  Return policy fn"
  [features weights possible-actions-fn]
  (fn [state]
    (let [actions               (possible-actions-fn state)
          state-action-data     (map (fn [action] 
                                       {:old-state state
                                        :action action})
                                     actions)
          feature-values        (feature-matrix features state-action-data)
          mult-with-weights     (fn [feat-val]
                                  (mmul feat-val weights))
          scores                (pmap mult-with-weights feature-values)
          values                (map vector scores actions)
          sorted                (sort-by first > values)]
      (-> sorted first second))))

(defn transition
  "Generate a data point by making a state transition."
  [policy-fn reward-fn transition-fn state]
  (let [action     (policy-fn state)
        new-state  (transition-fn state action)]
    {:old-state state
     :action    action
     :reward    (reward-fn new-state action)
     :new-state new-state}))

(defn trajectory
  "Generate a trajectory by following a policy defined by weights."
  [max-length
   goal-state
   policy-fn
   reward-fn
   transition-fn
   initial-state]
  (let [termination?  (fn [samples]
                        (or (= (count samples) max-length)
                            (= (-> samples last :new-state) goal-state)))

        trans-fn      (partial transition
                               policy-fn
                               reward-fn
                               transition-fn)]
    (loop [samples  [(trans-fn initial-state)]]
      (if (termination? samples)
        samples
        (recur (conj samples
                     (trans-fn (-> samples
                                   last
                                   :new-state))))))))

(defn iterate-a-and-b
  [training-data features weights discount possible-actions-fn]
  (loop [A      (zero-matrix (count features)
                             (count features))
         b      (zero-vector (count features))
         data   training-data]
    (if (zero? (count data))
      [A b]
      (recur (update-a A
                       features
                       (policy-action features
                                      weights
                                      possible-actions-fn)
                       discount
                       (first data))
             (update-b b
                       features
                       (first data))
             (rest data)))))



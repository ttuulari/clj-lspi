(ns clj_lspi.lsq-test
  (:refer-clojure :exclude [* - + == /])
  (:require
    [clojure.test :refer [deftest testing is]]
    [clojure.core.matrix :refer :all]
    [clojure.core.matrix.operators :refer :all]
    [clj-lspi.lsq :refer :all]))

(def training-data 
  [{:old-state 1 :new-state 4 :action 1 :reward 0}
   {:old-state 2 :new-state 5 :action 1 :reward 0}
   {:old-state 3 :new-state 6 :action 2 :reward 1}
   {:old-state 1 :new-state 4 :action 1 :reward 1}])

(def features
  [(fn [[state action]]
     (+ state action))
   
   (fn [[state action]]
     (- action state))
   
   (fn [[state action]]
     1)])

(defn test-actions
  [state]
  [0 1 2])

(deftest b-update-zero-reward
  (testing "b-update with zero reward"
    (let [input (matrix [1 2 3])]
    (is (= (update-b input
                     features
                     {:old-state 1 :new-state 4 :action 1 :reward 0})
           input)))))

(deftest b-update-one-reward
  (testing "b-update with reward of one"
    (let [input                 (matrix [1 2 3])
          sample                {:old-state 1 :new-state 4 :action 1 :reward 1}
          extract-feature-data  (fn [elem]
                                  [(:old-state elem)
                                   (:action elem)])
          feature-data          (extract-feature-data sample)
          feature-values        ((apply juxt features) feature-data)]
      (is (= (update-b input
                       features
                       sample)
             (+ input
                feature-values))))))

(deftest pick-best-action
  (testing "Pick the best action"
    (let [state                 1
          weights               [0.333 0.333 0.333]
          actions               (test-actions state)
          state-action-data     (map (fn [action] 
                                       {:old-state state
                                        :action action})
                                     actions)
          feat-mat              (feature-matrix features
                                                state-action-data)


          values                (map-indexed (fn [idx vals] [idx (reduce + vals)])
                                             feat-mat)
          sorted                (sort-by second > values)
          best-fn               (policy-action features
                                             weights
                                             test-actions)
          best                  (best-fn state)]
      (is (= (-> sorted first first) 
             best)))))



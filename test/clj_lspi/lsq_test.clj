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

(def test-features
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
                     test-features
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
          feature-values        ((apply juxt test-features) feature-data)]
      (is (= (update-b input
                       test-features
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
          feat-mat              (feature-matrix test-features
                                                state-action-data)


          values                (map-indexed (fn [idx vals] [idx (reduce + vals)])
                                             feat-mat)
          sorted                (sort-by second > values)
          best-fn               (policy-action test-features
                                               weights
                                               test-actions)
          best                  (best-fn state)]
      (is (= (-> sorted first first) 
             best)))))

;;;;;;;;;;;; Tests for learning to do addition ;;;;;;;;;;;;;;;;;;;;;;;;

(def goal              50)
(def discount          0.9)
(def reward            (fn [s]
                            (/ 1 (+ 0.01 (Math/abs (- goal s))))))
(def starting-states   (range 0 (* goal 2)))
(def possible-actions  (fn [s]
                            (cond
                              (= goal s) [0 1 -1]
                              :else (range (* -1 (/ goal 2)) (/ goal 2)))))
(def mapper            (fn [s]
                            (let [action  (rand-nth (possible-actions s))]
                              {:old-state s
                               :action    action 
                               :reward    (reward (+ s action))
                               :new-state (+ s action)})))

(def training-data     (loop [samples  [(mapper 0)]]
                            (-> samples last :new-state)
                            (if (or (= (count samples) 100)
                                    (=  (-> samples last :new-state) goal))
                              samples
                              (recur (conj samples
                                           (mapper (:new-state (last samples))))))))

(def features          [(fn [[s a]] s)
                        (fn [[s a]] a)
                        (fn [[s a]] (- goal s))
                        (fn [[s a]] (if (pos? s) 1 0))])
(def f-c               (count features))
(def init-weights      (repeat f-c (/ 1 f-c)))

(deftest addition-test
  (testing "Testing addition"
    (let [start-params      (a-and-b features
                                     training-data
                                     (policy-action features
                                                    init-weights
                                                    possible-actions) 
                                     discount)]
      (is (matrix? (:a start-params)))
      (is (vec?    (:b start-params))))))

(deftest weights-test
  (testing "Solve weights"
    (let [w   (solve-weights features
                             training-data
                             (policy-action features
                                            init-weights
                                            possible-actions)
                             discount)]
      (is (vec? w)))))

(deftest weights-test-2
  (testing "Solve weights twice"
    (let [w   (solve-weights features
                             training-data
                             (policy-action features
                                            init-weights
                                            possible-actions)
                             discount)
          w2  (solve-weights features
                             training-data
                             (policy-action features
                                            w
                                            possible-actions)
                             discount)]
      (is (vec? w))
      (is (vec? w2))
      (is (zero? (distance w w2))))))


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

(def goal              10)
(def discount          0.9)
(defn reward
  [state _action]
  (/ 10 (+ 0.01 (Math/abs (- goal state)))))

(defn possible-actions
  [state]
  (cond
    (= goal state) [0]
    :else (range (* -1 (/ goal 2)) (/ goal 2))))

(defn transition-fn
  [state action]
  (+ state action))

(def features          [(fn [[s a]] s)
                        (fn [[s a]] a)
                        (fn [[s a]] (- goal s))])

(def init-weights    (repeat (count features) 0))

(defn random-policy
  [state]
  (rand-nth (possible-actions state)))

(defn random-training-data
  [length]
  (trajectory length
              goal
              random-policy
              reward 
              transition-fn
              (rand-int 100)))

(def training-data (apply concat (pmap random-training-data (repeat 200 20))))
(println training-data)

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

(deftest a-update
  (testing "Update A matrix"
    (let [a-result (loop [A      (zero-matrix (count features)
                                              (count features))
                          data   training-data]
                     (if (zero? (count data))
                       A
                       (recur (update-a A
                                        features
                                        (policy-action features
                                                       init-weights
                                                       possible-actions)
                                        discount
                                        (first data))
                              (rest data))))]
      (is (matrix? a-result)))))

(deftest iterate-ab
  (testing "a and b update iteration"
    (let [[a b]  (iterate-a-and-b training-data features init-weights discount possible-actions)]
      (is (vec? b))
      (is matrix? a))))

(deftest iterate-w
  (testing "Iterate weights"
    (loop [counter  0
           w        init-weights]
      (println "w" w)
      (let [[a b]   (iterate-a-and-b training-data features w discount possible-actions)
            new-w   (weights a b)]
        (if (< counter 5)
          (recur (inc counter) new-w)
          new-w)))))


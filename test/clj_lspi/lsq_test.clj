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

(deftest addition-test
  (let [goal              50
        discount          0.9
        reward            (fn [s]
                            (/ 1 (+ 0.01 (Math/abs (- goal s)))))
        starting-states   (range 0 (* goal 2))
        possible-actions  (fn [s]
                            (cond
                              (= goal s) [0 1 -1]
                              :else (range (* -1 (/ goal 2)) (/ goal 2))))
        mapper            (fn [s]
                            (let [action  (rand-nth (possible-actions s))]
                              {:old-state s
                               :action    action 
                               :reward    (reward (+ s action))
                               :new-state (+ s action)}))

        training-data     (loop [samples  [(mapper 0)]]
                            (-> samples last :new-state)
                            (if (or (= (count samples) 100)
                                    (=  (-> samples last :new-state) goal))
                              samples
                              (recur (conj samples
                                           (mapper (:new-state (last samples)))))))


        features          [(fn [[s a]] s)
                           (fn [[s a]] a)
                           (fn [[s a]] (- goal s))
                           (fn [[s a]] (if (pos? s) 1 0))]
        f-c               (count features)
        init-weights      (repeat f-c (/ 1 f-c))]
    
    (testing "Testing addition"
      (let [start-params      (initial-params features
                                              training-data
                                              possible-actions
                                              discount
                                              init-weights)]
        (is (matrix? (:a start-params)))
        (is (vec?    (:b start-params)))
        (is (vec?    (:w start-params)))))

   (testing "Solve weights"
      (let [w   (solve-weights features
                               training-data
                               (policy-action features
                                              init-weights
                                              possible-actions)
                               discount)]
        (println "w" w)
        (is (vec? w))))))






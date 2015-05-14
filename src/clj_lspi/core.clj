(ns clj-lspi.core
  (:gen-class)
  (:refer-clojure :exclude [* - + == /])
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :refer :all]))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (let [mat (matrix [[1 2]
                     [3 4]])

        {:keys [U S V*]} (svd mat)]
    (pm U)))

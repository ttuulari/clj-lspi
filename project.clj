(defproject clj-lspi "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.7.0-beta2"]
                 [net.mikera/core.matrix "0.34.0"]
                 [org.clojure/data.json "0.2.6"]]

  :main ^:skip-aot clj-lspi.core 
  :target-path "target/%s"
  :source-paths ["src"] 
  :profiles {:uberjar {:aot :all}
             :dev {:plugins [[com.jakemccrary/lein-test-refresh "0.9.0"]
                             [lein-cloverage "1.0.6"]]}})

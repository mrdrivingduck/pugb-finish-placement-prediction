mvn compile
mvn exec:java -Dexec.mainClass="iot.zjt.pugb.util.svm_train" -Dexec.args="-s 4 -t 0 ./data/inputScaled.data ./model/nu-SVR-linear.model"  
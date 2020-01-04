package iot.zjt.pugb.preprocess;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.Reader;
import java.io.Writer;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

public class Format {

    private static String[] features = new String[] {
        // "assists",
        // "damageDealt",
        // "DBNOs",
        // "headshotKills",
        // "heals",
        // "killPlace",
        // "killPoints",
        // "kills",
        // "killStreaks",
        // "longestKill",
        // "maxPlace",
        // "rankPoints",
        // "revives",
        // "roadKills",
        // "teamKills",
        "walkDistance",
        // "swimDistance",
        // "weaponsAcquired",
        // "winPoints"
    };

    private static String labelFeature = new String("winPlacePerc");

    public static void main(String[] args) throws Exception {

        Writer out = new FileWriter("data/input.data");

        Reader in = new FileReader("data/train_V2.csv");
        Iterable<CSVRecord> recordIter = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(in);
        for (CSVRecord record : recordIter) {

            String label = record.get(labelFeature);
            if (label == null || label.equals("")) {
                label = new String("0.0000");
            }
            out.write(label);
            out.write(" ");

            for (int i = 0; i < features.length; i++) {
                out.write(Integer.toString(i + 1));
                out.write(":");
                out.write(record.get(features[i]));
                out.write(" ");
            }

            out.write('\n');
        }
        out.close();
    }
}
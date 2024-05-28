package org.example.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.example.model.StoreResult;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class StoreResultList {
    private static StoreResultList instance;

    private List<StoreResult> storeResultList;

    private StoreResultList() {
        storeResultList = new ArrayList<>();
    }

    public static StoreResultList getInstance() {
        if (instance == null) {
            instance = new StoreResultList();
        }
        return instance;
    }

    public void append(StoreResult result) {
        storeResultList.add(result);
    }

    public List<StoreResult> getStoreResultList() {
        return this.storeResultList;
    }

    public void writeToFile() {
        String filename = "result/result.json";
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String jsonString = gson.toJson(this.storeResultList);

        try {
            File file = new File(filename);
            if (!file.exists()) {
                file.createNewFile(); // Create the file if it doesn't exist
            }
            FileWriter writer = new FileWriter(file, false); // Setting append mode to false to overwrite existing content
            writer.write(jsonString);
            writer.close();
            System.out.println("JSON data has been written to file successfully.");
        } catch (IOException e) {
            System.out.println("An error occurred while writing to the file.");
            e.printStackTrace();
        }
    }
}
